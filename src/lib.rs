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
use std::time::SystemTime;

use ndarray::*;
use ndarray_linalg::Inverse;

use lv2::prelude::*;

const C: f32 = 343.00; /* m*s^-1 */

// Let's just hard-code this for now
const MIC_SPACING: f32 = 0.02; /* m */

fn to_radians(degrees: f32) -> f32 {
    degrees * (PI / 180f32)
}

/*
 * The steering vector
 */
fn build_steering_vector(angle: f32, opt_freq: f32) -> Array1<f32> {

    // Mic positions are relative to Left/Top to preserve x/y axis semantics
    let mic_positions: Array1<Array1<f32>> = ndarray::array![
        ndarray::array![0f32, 0f32],
        ndarray::array![MIC_SPACING, 0f32],
        ndarray::array![MIC_SPACING / 2f32, 3f32.sqrt() / (MIC_SPACING / 2f32)]
    ];

    // Derive the unit vector of the steering angle
    let unit_vector: Array1<f32> = ndarray::array![angle.cos(), angle.sin()];

    // Collect up the steering vector
    Array1::from_shape_vec(3,
        mic_positions.iter()
        .map(|mic| mic.dot(&unit_vector))
        .map(|delay| 2f32 * PI * opt_freq * (delay / C))
        .collect()).unwrap()
}

fn create_sample_array(inputs: &[&[f32]], num_samples: usize) -> Array2<f32> {
    let flat_array: Vec<f32> = inputs.iter()
                                     .flat_map(|&slice| slice.iter().copied())
                                     .collect();

    Array2::from_shape_vec((3, num_samples), flat_array)
        .expect("Could not create array from sample vector!")
}

fn build_covariance_matrix(signals: ArrayView2<f32>, num_samples: usize) -> Array2<f32> {
    let transposed = signals.t();

    let covar = signals.dot(&transposed) / num_samples as f32;

    covar
}

/*
 * Input and output ports used by the plugin
 *
 *
 * Ports:
 *      in_1: channel 1 input (vertex of triangle)
 *      in_2: channel 2 input (left/top)
 *      in_3: channel 3 input (right/bottom)
 *      out: output
 *      angle: steering angle in degrees
 *      opt_freq: frequency to optimise for
 *      t_win: covariance matrix time window
 */
#[derive(PortCollection)]
struct Ports {
    in_1: InputPort<Audio>,
    in_2: InputPort<Audio>,
    in_3: InputPort<Audio>,
    out: OutputPort<Audio>,
    angle: InputPort<Control>,
    opt_freq: InputPort<Control>,
    t_win: InputPort<Control>
}

/*
 * Plugin state
 */
#[uri("https://chadmed.au/triforce")]
struct Triforce {
    angle_curr: f32,
    last_update: SystemTime,
    freq_curr: f32,
    steering_vector: Array1<f32>,
    covar: Array2<f32>,
}

/*
 * Extend the Plugin trait so that we can modularly update the parameters of
 * the plugin IFF they have changed.
 */
trait Beamformer: Plugin {
    fn update_params(&mut self, ports: &mut Ports);
}

impl Plugin for Triforce {
    type Ports = Ports;

    type InitFeatures = ();
    type AudioFeatures = ();

    fn new(_info: &PluginInfo, _features: &mut ()) -> Option<Self> {
        Some(Self {
            angle_curr: 0f32,
            last_update: SystemTime::now(),
            freq_curr: 1000f32,
            steering_vector: build_steering_vector(to_radians(90f32), 1000f32),
            covar: ndarray::array!([], []),
        })
    }

    fn run(&mut self, ports: &mut Ports, _features: &mut (), _: u32) {
        Beamformer::update_params(self, ports);

        // Steering vector is relative to Left/Top mic
        let inputs = vec![
            *ports.in_2,
            *ports.in_1,
            *ports.in_3
        ];
        let num_samples = inputs[0].len();
        if num_samples <= 0 {
            return;
        }

        let samples = create_sample_array(&inputs, num_samples);

        // Do some time handling to determine if we need to update the covariance
        // matrix
        if self.last_update.elapsed().unwrap().as_millis() > *ports.t_win as u128 {
            self.covar = build_covariance_matrix(samples.view(), num_samples);
        }

        // We need to get the inverse of the covariance matrix, then get the dot product
        // of that and the steering vector. The covariance matrix is always square.
        let rv = self.covar
            .inv()
            .expect("Could not invert covariance matrix")
            .dot(&self.steering_vector);

        let norm = self.steering_vector.dot(&rv);

        let weights = rv / norm;

        let output = samples.t().dot(&weights);

        // Now make sure our output will fit in our LV2 buffer
        assert_eq!(output.len(), ports.out.len());

        for (i, &sample) in output.iter().enumerate() {
            ports.out[i] = sample;
        }
    }
}

impl Beamformer for Triforce {
    fn update_params(&mut self, ports: &mut Ports) {
        if self.angle_curr != *ports.angle ||
            self.freq_curr != *ports.opt_freq {
            self.angle_curr = *ports.angle;
            self.freq_curr = *ports.opt_freq;
            self.steering_vector = build_steering_vector(to_radians(self.angle_curr), self.freq_curr);
        }
    }
}

lv2_descriptors!(Triforce);
