// SPDX-License-Identifier: GPL-2.0-or-later
/*
 * An attempt at an MVDR beamformer for equilateral triangle mic arrays
 * as found on newer Apple Silicon Macs.
 *
 * Currently mono, but could probably be extended to be stereo somewhat
 * easily. I think.
 *
 * Copyright (C) 2024 James Calligeros <jcalligeros99@gmail.com>
 */

use std::f32::consts::PI;

use lv2::prelude::*;

use nalgebra::{Matrix3, Vector3};

use rustfft::{num_complex::Complex, num_traits::Zero, FftPlanner};

use clarabel::{algebra::*, solver::*};

const C: f32 = 343.00; /* m*s^-1 */

/* Robustness parameter
* Setting to zero exactly matches behavior of
* non-robust beamformer, setting to higher values
* will be more robust to errors in steering vector */
const R: f64 = 0.1;

/* Diagonal loading constant for covariance matrix
* This should be as small as possible to to prevent
* numerical issues from a near-zero covariance eigenvalue */
const C_DL: f64 = 1e-5;

/// The distance of a given element in the array from the zeroth
/// element
#[derive(Copy, Clone)]
struct ElemDistance {
    x: f32,
    y: f32,
}

/// Perform a Hilbert transform on a slice of f32s to give us the analytic signal of
/// our input sample buffer. This is necessary to extract phase information from the
/// signal, and to make matrix operations a bit easier.
fn analytic_signal(
    planner: &mut FftPlanner<f32>,
    signal: &[f32],
    len: usize,
    output: &mut Vec<Complex<f32>>,
    fft_scratch: &mut Vec<Complex<f32>>,
) {
    // Convert each real sample into a complex sample
    output.resize(len, Complex::zero());
    for (o, i) in output.iter_mut().zip(signal.iter()) {
        *o = Complex::new(*i, 0.0);
    }

    // Set up the fft and inverse fft
    let fft = planner.plan_fft_forward(len);
    let ifft = planner.plan_fft_inverse(len);

    // Mutate the output buffer into the forward FFT
    fft.process_with_scratch(output, fft_scratch);

    // Perform the Hilbert transform on the FFT. To do this, we multiply every
    // positive sample under the Nyquist limit by 2+0j, and destroy every sample
    // above it.
    for i in 0..len {
        if i > 0 && i < len / 2 {
            output[i] *= Complex::new(2.0, 0.0);
        } else if i >= len / 2 {
            output[i] = Complex::zero();
        }
    }

    // Turn the original complex buffer into the inverse FFT and then normalise
    ifft.process_with_scratch(output, fft_scratch);
    output.iter_mut().for_each(|x| *x /= len as f32);
}

/// The steering vector is a representation of the phase delays at each microphone.
/// It is calculated by taking the dot product of the array geometry matrix and the
/// unit vector of the direction of arrival.
fn steering_vec(theta: f32, phi: f32, f: f32, elems: [ElemDistance; 3]) -> Vector3<Complex<f32>> {
    // Mic positions are relative to Left/Top to preserve x/y axis semantics
    let mic_positions: Vec<Vector3<f32>> =
        elems.iter().map(|e| Vector3::new(e.x, e.y, 0f32)).collect();

    // Calculate angular repetency (2pi/lambda)
    let repetency = (2f32 * PI) / (f / C);

    // Compute the unit vector of the DOA
    let u_dir = Vector3::new(phi.sin() * theta.cos(), phi.sin() * theta.sin(), phi.cos());

    // Calculate the steering vector by taking the array geometry, speed of sound,
    // and DOA unit vector
    let mut steering_vector = Vector3::from_element(Complex::new(0f32, 0f32));

    for (i, mic_pos) in mic_positions.iter().enumerate() {
        let delay = mic_pos.dot(&u_dir) / C;
        let phase = -repetency * delay;
        steering_vector[i] = Complex::new(phase.cos(), phase.sin());
    }

    steering_vector
}

/// There's nothing special about this, it's just a covariance matrix. It is always
/// square.
fn covariance(signals: &[Vec<Complex<f32>>], n_samples: usize) -> Matrix3<Complex<f64>> {
    let mut covar = Matrix3::zeros();

    for t in 0..n_samples {
        let discrete: Vector3<Complex<f64>> = Vector3::from_iterator(
            signals
                .iter()
                .map(|s| Complex::new(s[t].re as f64, s[t].im as f64)),
        );
        covar += &discrete * discrete.adjoint();
    }
    covar /= Complex::new(n_samples as f64, 0f64);
    covar
}

fn mvdr_weights(cov: &Matrix3<Complex<f64>>, sv: &Vector3<Complex<f32>>) -> Vector3<Complex<f32>> {
    // J is 3x3 exchange matrix
    const J: Matrix3<Complex<f64>> = Matrix3::new(
        Complex::new(0.0, 0.0),
        Complex::new(0.0, 0.0),
        Complex::new(1.0, 0.0),
        Complex::new(0.0, 0.0),
        Complex::new(1.0, 0.0),
        Complex::new(0.0, 0.0),
        Complex::new(1.0, 0.0),
        Complex::new(0.0, 0.0),
        Complex::new(0.0, 0.0),
    );

    // Covariance matrix from forward-backward averaging,
    // which empirically reduces correlation between signals
    let cov_fba = (cov + J * cov.conjugate() * J).scale(0.5)
        + Matrix3::<Complex<f64>>::from_diagonal_element(Complex::<f64>::new(C_DL, 0.0));

    return match mvdr_weights_socp(&cov_fba, &sv, R) {
        Some(v) => v,
        None => Vector3::zeros(),
    };
}

/// Robust MVDR weights using second-order cone programming (SOCP)
#[allow(non_snake_case)]
fn mvdr_weights_socp(
    cov: &Matrix3<Complex<f64>>,
    sv: &Vector3<Complex<f32>>,
    eps: f64,
) -> Option<Vector3<Complex<f32>>> {
    const N: usize = 3;
    const N2: usize = 2 * N; // 6
    const NX: usize = N2 + 1; // w(6) + tau(1) = 7
    const M: usize = 2 * (N2 + 1) + 1; // Q^7 + Q^7 + Zero^1 = 15

    let mut sv_re = [0.0f64; N];
    let mut sv_im = [0.0f64; N];
    for k in 0..N {
        sv_re[k] = sv[k].re as f64;
        sv_im[k] = sv[k].im as f64;
    }

    // ---- Cholesky and real/imag blocks of U = L^H ----
    let chol = match cov.cholesky() {
        Some(m) => m,
        None => return None,
    };

    let U = chol.l().adjoint();
    let mut Ur = [[0.0f64; N]; N];
    let mut Ui = [[0.0f64; N]; N];
    for i in 0..N {
        for j in 0..N {
            Ur[i][j] = U[(i, j)].re;
            Ui[i][j] = U[(i, j)].im;
        }
    }

    // a = [Re(sv); Im(sv)],  abar = [Im(sv); -Re(sv)]
    let mut a = [0.0f64; N2];
    let mut abar = [0.0f64; N2];
    for k in 0..N {
        a[k] = sv_re[k];
        a[N + k] = sv_im[k];
        abar[k] = sv_im[k];
        abar[N + k] = -sv_re[k];
    }

    // ---- Build A x + s = b  ----
    let mut I: Vec<usize> = Vec::with_capacity(55);
    let mut J: Vec<usize> = Vec::with_capacity(55);
    let mut V: Vec<f64> = Vec::with_capacity(55);
    let mut b: Vec<f64> = vec![0.0; M];

    // SOC #1: [tau; U' w] ∈ Q^7  -> s1 = L1 x, use A1 = -L1, b1 = 0
    // row 0: pick tau
    I.push(0);
    J.push(N2);
    V.push(-1.0);
    // rows 1..6: -U' * w
    for i in 0..N2 {
        for j in 0..N2 {
            let val = if i < N && j < N {
                -Ur[i][j]
            } else if i < N && j >= N {
                -(-Ui[i][j - N])
            } else if i >= N && j < N {
                -(Ui[i - N][j])
            } else {
                -Ur[i - N][j - N]
            };
            if val != 0.0 {
                I.push(1 + i);
                J.push(j);
                V.push(val);
            }
        }
    }

    // SOC #2: [a^T w - 1; eps * w] ∈ Q^7
    let base = N2 + 1; // 7
    b[base] = -1.0; // constant in first component
                    // top row: -a^T w
    for j in 0..N2 {
        let aj = a[j];
        if aj != 0.0 {
            I.push(base);
            J.push(j);
            V.push(-aj);
        }
    }
    // rows base+1..base+6: -eps * w_i
    for i in 0..N2 {
        I.push(base + 1 + i);
        J.push(i);
        V.push(-eps);
    }

    // Equality: abar^T w = 0  -> Zero cone row
    let eq_row = M - 1; // 14
    for j in 0..N2 {
        let cj = abar[j];
        if cj != 0.0 {
            I.push(eq_row);
            J.push(j);
            V.push(cj);
        }
    }

    let A = CscMatrix::new_from_triplets(M, NX, I, J, V);

    // P = 0, q selects tau
    let P = CscMatrix::<f64>::zeros((NX, NX));
    let q = {
        let mut q = vec![0.0f64; NX];
        q[NX - 1] = 1.0;
        q
    };

    let cones = [
        SecondOrderConeT(N2 + 1),
        SecondOrderConeT(N2 + 1),
        ZeroConeT(1),
    ];

    let settings = DefaultSettingsBuilder::default()
        .verbose(false)
        .build()
        .unwrap();

    let mut solver =
        DefaultSolver::new(&P, &q, &A, &b, &cones, settings).expect("Could not create solver");

    solver.solve();

    return match solver.solution.status {
        clarabel::solver::SolverStatus::Solved => {
            let x = &solver.solution.x;
            Some(Vector3::new(
                Complex::new(x[0] as f32, x[3] as f32),
                Complex::new(x[1] as f32, x[4] as f32),
                Complex::new(x[2] as f32, x[5] as f32),
            ))
        }
        _ => None,
    };
}

/*
 * Input and output ports used by the plugin
 *
 *
 * Ports:
 *      in_1: channel 1 input (left/top)
 *      in_2: channel 2 input (right/bottom)
 *      in_3: channel 3 input (vertex)
 *      out: output
 *      h_angle: horizontal steering angle in degrees (relative to input 1)
 *      v_angle: vertical steering angle in degrees
 *      opt_freq: frequency to optimise for
 *      t_win: covariance matrix time window, in milliseconds
 *      t_bf:  interval between beamforming updates, in milliseconds
 */
#[derive(PortCollection)]
pub struct Ports {
    in_1: InputPort<Audio>,
    in_2: InputPort<Audio>,
    in_3: InputPort<Audio>,
    out: OutputPort<Audio>,
    h_angle: InputPort<Control>,
    v_angle: InputPort<Control>,
    opt_freq: InputPort<Control>,
    t_win: InputPort<Control>,
    t_bf: InputPort<Control>,
    mic2_x: InputPort<Control>,
    mic2_y: InputPort<Control>,
    mic3_x: InputPort<Control>,
    mic3_y: InputPort<Control>,
}

/*
 * Plugin state
 */
#[uri("https://chadmed.au/triforce")]
pub struct Triforce {
    hangle_curr: f32,
    vangle_curr: f32,
    freq_curr: f32,
    sample_rate: f32,
    samples_since_last_update: usize,
    covar_rbuffer: Vec<Matrix3<Complex<f64>>>,
    covar_rbuffer_i: usize,
    covar_rbuffer_sz: usize,
    steering_vector: Vector3<Complex<f32>>,
    covar: Matrix3<Complex<f64>>,
    array_geom: [ElemDistance; 3],
    fft_planner: FftPlanner<f32>,
    weights: Vector3<Complex<f32>>,
    inputs: [Vec<Complex<f32>>; 3],
    fft_scratch: Vec<Complex<f32>>,
    buf_len: usize,

    // Parameters
    t_win: f32, // Time window to accumulate covariance, in milliseconds
    t_bf: f32,  // Interval between beamforming weights updates, in milliseconds
}

trait Beamformer: Plugin {
    fn update_params(&mut self, ports: &mut Ports);
}

impl Triforce {
    pub fn with_sample_rate(sample_rate: f32) -> Self {
        Self {
            hangle_curr: 0f32,
            vangle_curr: 0f32,
            freq_curr: 1000f32,
            samples_since_last_update: 0,
            sample_rate,
            covar_rbuffer: vec![Matrix3::zeros(); 0],
            covar_rbuffer_i: 0,
            covar_rbuffer_sz: 0,
            array_geom: [ElemDistance { x: 0f32, y: 0f32 }; 3],
            steering_vector: steering_vec(
                90f32.to_radians(),
                45f32.to_radians(),
                1000f32,
                [ElemDistance { x: 0f32, y: 0f32 }; 3],
            ),
            covar: Matrix3::zeros(),
            fft_planner: FftPlanner::new(),
            weights: Vector3::zeros(),
            inputs: [Vec::new(), Vec::new(), Vec::new()],
            fft_scratch: Vec::with_capacity(0),
            buf_len: 0,

            // Sensible defaults: 2.5s covariance
            // buffer with beamforming every 1s.
            // Will be used only if not configured
            // through update_params().
            t_win: 2500.0,
            t_bf: 1000.0,
        }
    }

    pub fn process_slice(
        &mut self,
        mic1: &[f32],
        mic2: &[f32],
        mic3: &[f32],
        output: &mut [f32],
        buf_len: usize,
    ) {
        // If buffer length changed, update buffer sizes
        if buf_len != self.buf_len {
            self.buf_len = buf_len;
            self.expand_buffers();
        }

        // Steering vector is relative to Left/Top mic
        analytic_signal(
            &mut self.fft_planner,
            mic1,
            buf_len,
            &mut self.inputs[0],
            &mut self.fft_scratch,
        );
        analytic_signal(
            &mut self.fft_planner,
            mic2,
            buf_len,
            &mut self.inputs[1],
            &mut self.fft_scratch,
        );
        analytic_signal(
            &mut self.fft_planner,
            mic3,
            buf_len,
            &mut self.inputs[2],
            &mut self.fft_scratch,
        );

        // Update the covariance ringbuffer every tick
        self.covar_rbuffer[self.covar_rbuffer_i] = covariance(&self.inputs, buf_len);
        self.covar_rbuffer_i = (self.covar_rbuffer_i + 1) % self.covar_rbuffer_sz;

        // Update the weights
        if self.samples_since_last_update as f32
            >= (self.t_bf / (1000f32 * 1000f32)) * self.sample_rate
        {
            self.samples_since_last_update = 0;

            // Covariance is average of the covariances over the buffer.
            self.covar.set_zero();
            for i in 0..self.covar_rbuffer_sz {
                self.covar += self.covar_rbuffer[i];
            }

            self.covar.scale_mut(1f64 / self.covar_rbuffer_sz as f64);
            self.weights = mvdr_weights(&self.covar, &self.steering_vector);
        } else {
            self.samples_since_last_update += buf_len;
        }

        for t in 0..buf_len {
            let discrete: Vector3<Complex<f32>> =
                Vector3::from_iterator(self.inputs.iter().map(|s| s[t]));

            let out =
                // Conjugate-linear dot product
                self.weights.dotc(&discrete)
                // // Now we need to revert the Hilbert transform and output the signal
                .re;

            // Do all of our NFP and clamping here
            output[t] = if out.is_finite() && !out.is_nan() {
                out.clamp(-10f32, 10f32)
            } else {
                0f32
            };
        }
    }

    // Make sure buffers are big enough
    // Run whenever any of the following are updated:
    // buf_len, sample_rate, t_win
    fn expand_buffers(&mut self) {
        // Make sure that your buffers are big enough
        if self.buf_len > self.fft_scratch.len() {
            self.fft_scratch.resize(self.buf_len, Complex::zero());
        }

        self.covar_rbuffer_sz =
            (self.t_win / 1000.0 * self.sample_rate / (self.buf_len as f32)) as usize;

        if self.covar_rbuffer_sz > self.covar_rbuffer.len() {
            self.covar_rbuffer
                .resize(self.covar_rbuffer_sz, Matrix3::zeros());
        }

        self.covar_rbuffer_i = self.covar_rbuffer_i % self.covar_rbuffer_sz;
    }
}

impl Plugin for Triforce {
    type Ports = Ports;

    type InitFeatures = ();
    type AudioFeatures = ();

    fn new(info: &PluginInfo, _features: &mut ()) -> Option<Self> {
        Some(Self::with_sample_rate(info.sample_rate() as f32))
    }

    fn run(&mut self, ports: &mut Ports, _features: &mut (), samples: u32) {
        Beamformer::update_params(self, ports);
        self.process_slice(
            &ports.in_1,
            &ports.in_2,
            &ports.in_3,
            &mut ports.out,
            samples as usize,
        );
    }
}

impl Beamformer for Triforce {
    fn update_params(&mut self, ports: &mut Ports) {
        if self.hangle_curr != *ports.h_angle
            || self.freq_curr != *ports.opt_freq
            || self.vangle_curr != *ports.v_angle
            || self.array_geom[1].x != *ports.mic2_x
            || self.array_geom[1].y != *ports.mic2_y
            || self.array_geom[2].x != *ports.mic3_x
            || self.array_geom[2].y != *ports.mic3_y
            || self.t_win != *ports.t_win
            || self.t_bf != *ports.t_bf
        {
            self.hangle_curr = *ports.h_angle;
            self.vangle_curr = *ports.v_angle;
            self.freq_curr = *ports.opt_freq;
            self.array_geom = [
                ElemDistance { x: 0f32, y: 0f32 },
                ElemDistance {
                    x: *ports.mic2_x,
                    y: *ports.mic2_y,
                },
                ElemDistance {
                    x: *ports.mic3_x,
                    y: *ports.mic3_y,
                },
            ];
            self.steering_vector = steering_vec(
                self.hangle_curr.to_radians(),
                self.vangle_curr.to_radians(),
                self.freq_curr,
                self.array_geom,
            );

            self.t_win = *ports.t_win;
            self.t_bf = *ports.t_bf;

            // We need to wait until we see a buffer to
            // know how much space to allocate.
            if self.buf_len > 0 {
                self.expand_buffers();
            }
        }
    }
}

lv2_descriptors!(Triforce);
