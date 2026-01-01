//! Integration tests for ruvector-learning-wasm
//!
//! Tests for adaptive learning mechanisms:
//! - MicroLoRA: Lightweight Low-Rank Adaptation
//! - Operator-scoped LoRA adapters
//! - Trajectory tracking

#[cfg(test)]
mod tests {
    use ruvector_learning_wasm::{LoRAConfig, LoRAPair, MicroLoRAEngine};

    // ========================================================================
    // MicroLoRA Tests
    // ========================================================================

    #[test]
    fn test_micro_lora_initialization() {
        let config = LoRAConfig::default();
        let lora = LoRAPair::new(&config);

        // Verify initial state
        assert_eq!(lora.adapt_count(), 0);
        assert_eq!(lora.param_count(), 256 * 2 + 256 * 2); // A and B matrices
    }

    #[test]
    fn test_micro_lora_forward_pass() {
        let config = LoRAConfig::default();
        let lora = LoRAPair::new(&config);

        let input: Vec<f32> = vec![1.0; 256];
        let output = lora.forward(&input);

        assert_eq!(output.len(), 256);

        // Initially B is zeros, so output should equal input
        for i in 0..256 {
            assert!(
                (output[i] - input[i]).abs() < 1e-6,
                "Initial output should match input at index {}: {} vs {}",
                i,
                output[i],
                input[i]
            );
        }
    }

    #[test]
    fn test_micro_lora_forward_into() {
        let config = LoRAConfig::default();
        let lora = LoRAPair::new(&config);

        let input: Vec<f32> = vec![0.5; 256];
        let mut output = vec![0.0; 256];

        lora.forward_into(&input, &mut output);

        // Initially should match input
        for i in 0..256 {
            assert!(
                (output[i] - input[i]).abs() < 1e-6,
                "forward_into output should match input"
            );
        }
    }

    #[test]
    fn test_micro_lora_adaptation() {
        let config = LoRAConfig::default();
        let mut lora = LoRAPair::new(&config);

        let gradient: Vec<f32> = vec![0.1; 256];
        lora.adapt(&gradient);

        assert_eq!(lora.adapt_count(), 1);
        assert!(
            lora.delta_norm() > 0.0,
            "Delta norm should be positive after adaptation"
        );
    }

    #[test]
    fn test_micro_lora_forward_after_adaptation() {
        let config = LoRAConfig::default();
        let mut lora = LoRAPair::new(&config);

        // Adapt with gradient
        let gradient: Vec<f32> = vec![0.1; 256];
        lora.adapt(&gradient);

        // Forward should now produce different output
        let input: Vec<f32> = vec![1.0; 256];
        let output = lora.forward(&input);

        // Output should differ from input after adaptation
        let diff: f32 = input
            .iter()
            .zip(output.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();

        assert!(
            diff > 0.0,
            "Output should differ from input after adaptation"
        );
    }

    #[test]
    fn test_micro_lora_adapt_with_reward() {
        let config = LoRAConfig::default();
        let mut lora = LoRAPair::new(&config);

        let gradient: Vec<f32> = vec![0.1; 256];

        // Positive reward should trigger adaptation
        lora.adapt_with_reward(&gradient, 1.0);
        assert_eq!(lora.adapt_count(), 1);

        // Zero/negative reward should not adapt
        let initial_norm = lora.delta_norm();
        lora.adapt_with_reward(&gradient, 0.0);
        assert_eq!(lora.adapt_count(), 1); // No change
        lora.adapt_with_reward(&gradient, -1.0);
        assert_eq!(lora.adapt_count(), 1); // No change

        // Positive reward should adapt again
        lora.adapt_with_reward(&gradient, 0.5);
        assert_eq!(lora.adapt_count(), 2);
        assert!(lora.delta_norm() > initial_norm);
    }

    #[test]
    fn test_micro_lora_reset() {
        let config = LoRAConfig::default();
        let mut lora = LoRAPair::new(&config);

        // Adapt multiple times
        let gradient: Vec<f32> = vec![0.1; 256];
        lora.adapt(&gradient);
        lora.adapt(&gradient);
        lora.adapt(&gradient);

        assert!(lora.adapt_count() > 0);
        assert!(lora.delta_norm() > 0.0);

        // Reset
        lora.reset();

        assert_eq!(lora.adapt_count(), 0);
        assert!(
            lora.delta_norm() < 1e-10,
            "Delta norm should be ~0 after reset"
        );
    }

    #[test]
    fn test_micro_lora_parameter_efficiency() {
        let config = LoRAConfig {
            dim: 256,
            rank: 2,
            alpha: 0.1,
            learning_rate: 0.01,
            dropout: 0.0,
        };
        let lora = LoRAPair::new(&config);

        // LoRA params: A (256x2) + B (2x256) = 1024 params
        // Full matrix: 256x256 = 65536 params
        let lora_params = lora.param_count();
        let full_params = 256 * 256;

        assert!(
            lora_params < full_params / 10,
            "LoRA should use 10x fewer params: {} vs {}",
            lora_params,
            full_params
        );
    }

    // ========================================================================
    // MicroLoRA Engine Tests
    // ========================================================================

    #[test]
    fn test_engine_creation() {
        let engine = MicroLoRAEngine::default();
        let (forwards, adapts, delta) = engine.stats();

        assert_eq!(forwards, 0);
        assert_eq!(adapts, 0);
        assert!(delta < 1e-10);
    }

    #[test]
    fn test_engine_forward_and_adapt() {
        let mut engine = MicroLoRAEngine::default();

        let input: Vec<f32> = vec![1.0; 256];
        let _ = engine.forward(&input);
        engine.adapt(&input);

        let (forwards, adapts, delta) = engine.stats();
        assert_eq!(forwards, 1);
        assert_eq!(adapts, 1);
        assert!(delta >= 0.0);
    }

    #[test]
    fn test_engine_multiple_forward_passes() {
        let mut engine = MicroLoRAEngine::default();

        let input: Vec<f32> = vec![0.5; 256];

        // Multiple forward passes
        for _ in 0..10 {
            let _ = engine.forward(&input);
        }

        let (forwards, adapts, _) = engine.stats();
        assert_eq!(forwards, 10);
        assert_eq!(adapts, 0);
    }

    #[test]
    fn test_engine_reset() {
        let mut engine = MicroLoRAEngine::default();

        let input: Vec<f32> = vec![1.0; 256];
        let _ = engine.forward(&input);
        engine.adapt(&input);

        engine.reset();

        let (forwards, adapts, delta) = engine.stats();
        assert_eq!(forwards, 0);
        assert_eq!(adapts, 0);
        assert!(delta < 1e-10);
    }

    #[test]
    fn test_engine_lora_access() {
        let engine = MicroLoRAEngine::default();

        let lora = engine.lora();
        assert_eq!(lora.adapt_count(), 0);
    }

    // ========================================================================
    // Numerical Stability Tests
    // ========================================================================

    #[test]
    fn test_numerical_stability_zero_gradient() {
        let config = LoRAConfig::default();
        let mut lora = LoRAPair::new(&config);

        // Zero gradient should not cause issues
        let zero_gradient: Vec<f32> = vec![0.0; 256];
        lora.adapt(&zero_gradient);

        // Should still be able to forward
        let input: Vec<f32> = vec![1.0; 256];
        let output = lora.forward(&input);

        // All values should be finite
        for val in &output {
            assert!(val.is_finite(), "Output should be finite");
        }
    }

    #[test]
    fn test_numerical_stability_many_adaptations() {
        let config = LoRAConfig::default();
        let mut lora = LoRAPair::new(&config);

        let gradient: Vec<f32> = vec![0.01; 256];

        // Many adaptations
        for _ in 0..1000 {
            lora.adapt(&gradient);
        }

        // Should still produce finite outputs
        let input: Vec<f32> = vec![1.0; 256];
        let output = lora.forward(&input);

        for val in &output {
            assert!(
                val.is_finite(),
                "Output should remain finite after many adaptations"
            );
        }
    }

    #[test]
    fn test_numerical_stability_large_values() {
        let config = LoRAConfig::default();
        let mut lora = LoRAPair::new(&config);

        // Large gradient values
        let large_gradient: Vec<f32> = vec![1000.0; 256];
        lora.adapt(&large_gradient);

        let input: Vec<f32> = vec![1.0; 256];
        let output = lora.forward(&input);

        for val in &output {
            assert!(
                val.is_finite(),
                "Output should be finite even with large gradients"
            );
        }
    }

    #[test]
    fn test_numerical_stability_varied_input() {
        let config = LoRAConfig::default();
        let lora = LoRAPair::new(&config);

        // Varied input values
        let input: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) / 128.0).collect();
        let output = lora.forward(&input);

        for (i, val) in output.iter().enumerate() {
            assert!(
                val.is_finite(),
                "Output at {} should be finite, got {}",
                i,
                val
            );
        }
    }

    // ========================================================================
    // Custom Config Tests
    // ========================================================================

    #[test]
    fn test_custom_lora_config() {
        let config = LoRAConfig {
            dim: 128,
            rank: 2,
            alpha: 0.5,
            learning_rate: 0.1,
            dropout: 0.0,
        };
        let lora = LoRAPair::new(&config);

        let input: Vec<f32> = vec![1.0; 128];
        let output = lora.forward(&input);

        assert_eq!(output.len(), 128);
    }

    #[test]
    fn test_small_dimension_lora() {
        let config = LoRAConfig {
            dim: 32,
            rank: 2,
            alpha: 0.1,
            learning_rate: 0.01,
            dropout: 0.0,
        };
        let mut lora = LoRAPair::new(&config);

        let input: Vec<f32> = vec![0.5; 32];
        let gradient: Vec<f32> = vec![0.1; 32];

        // Forward
        let output1 = lora.forward(&input);
        assert_eq!(output1.len(), 32);

        // Adapt
        lora.adapt(&gradient);

        // Forward after adapt
        let output2 = lora.forward(&input);
        assert_eq!(output2.len(), 32);

        // Should be different
        let diff: f32 = output1
            .iter()
            .zip(output2.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 0.0, "Output should change after adaptation");
    }
}
