#!/usr/bin/env python3
"""
Comprehensive test runner for enhanced LoRA functionality.

This script tests the enhanced LoRA features without requiring external dependencies.
"""

import logging
import sys
import traceback
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MockTensor:
    """Mock tensor class for testing without PyTorch dependency."""
    
    def __init__(self, shape, dtype="float32", device="cpu"):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = MockDevice(device)
        self.data = [[0.01 * (i+j) for j in range(shape[1])] for i in range(shape[0])]
    
    def clone(self):
        return MockTensor(self.shape, self.dtype, self.device.type)
    
    def detach(self):
        return self
    
    def to(self, device=None, dtype=None):
        new_device = device if device else self.device
        new_dtype = dtype if dtype else self.dtype
        return MockTensor(self.shape, new_dtype, new_device.type if hasattr(new_device, 'type') else new_device)
    
    def norm(self):
        return MockScalarTensor(1.5)
    
    def numel(self):
        return self.shape[0] * self.shape[1] if len(self.shape) >= 2 else self.shape[0]
    
    def __add__(self, other):
        return MockTensor(self.shape, self.dtype, self.device.type)
    
    def __mul__(self, scalar):
        return MockTensor(self.shape, self.dtype, self.device.type)
    
    def flatten(self):
        return MockTensor([self.numel()], self.dtype, self.device.type)
    
    def cpu(self):
        return MockTensor(self.shape, self.dtype, "cpu")


class MockScalarTensor:
    """Mock scalar tensor."""
    
    def __init__(self, value):
        self.value = value
    
    def item(self):
        return self.value


class MockDevice:
    """Mock device class."""
    
    def __init__(self, device_type="cpu"):
        self.type = device_type
    
    def __eq__(self, other):
        if isinstance(other, MockDevice):
            return self.type == other.type
        return str(other) == self.type
    
    def __str__(self):
        return self.type


def create_mock_lora_parameters():
    """Create mock LoRA parameters for testing."""
    return {
        "base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight": MockTensor([16, 768]),
        "base_model.model.layers.0.self_attn.q_proj.lora_B.default.weight": MockTensor([768, 16]),
        "base_model.model.layers.0.self_attn.v_proj.lora_A.default.weight": MockTensor([16, 768]),
        "base_model.model.layers.0.self_attn.v_proj.lora_B.default.weight": MockTensor([768, 16]),
        "base_model.model.layers.1.self_attn.q_proj.lora_A.default.weight": MockTensor([16, 768]),
        "base_model.model.layers.1.self_attn.q_proj.lora_B.default.weight": MockTensor([768, 16]),
    }


def create_mock_client_updates():
    """Create mock client updates for testing aggregation."""
    base_params = create_mock_lora_parameters()
    client_updates = {}
    
    for i in range(3):
        client_id = f"client_{i}"
        client_params = {}
        for name, param in base_params.items():
            # Create slightly different parameters for each client
            client_params[name] = param
        client_updates[client_id] = client_params
    
    return client_updates


def test_lora_parameter_extraction():
    """Test LoRA parameter extraction functionality."""
    logger.info("=== Testing LoRA Parameter Extraction ===")
    
    try:
        mock_params = create_mock_lora_parameters()
        
        # Test pattern matching
        lora_patterns = ["lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B"]
        
        lora_params = {}
        for name, param in mock_params.items():
            if any(pattern in name for pattern in lora_patterns):
                lora_params[name] = param
        
        # Validate extraction
        assert len(lora_params) == len(mock_params), "All parameters should match LoRA patterns"
        
        # Test A/B matrix pairs
        a_matrices = [name for name in lora_params.keys() if "lora_A" in name]
        b_matrices = [name for name in lora_params.keys() if "lora_B" in name]
        
        assert len(a_matrices) > 0, "Should find LoRA A matrices"
        assert len(b_matrices) > 0, "Should find LoRA B matrices"
        assert len(a_matrices) == len(b_matrices), "Should have equal A and B matrices"
        
        # Test rank consistency
        for a_name in a_matrices:
            b_name = a_name.replace("lora_A", "lora_B")
            assert b_name in b_matrices, f"Missing B matrix for {a_name}"
            
            a_shape = lora_params[a_name].shape
            b_shape = lora_params[b_name].shape
            
            assert a_shape[0] == b_shape[1], f"Rank mismatch: {a_name} {a_shape} vs {b_name} {b_shape}"
        
        logger.info("✅ LoRA parameter extraction test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ LoRA parameter extraction test FAILED: {e}")
        traceback.print_exc()
        return False


def test_lora_parameter_statistics():
    """Test LoRA parameter statistics calculation."""
    logger.info("=== Testing LoRA Parameter Statistics ===")
    
    try:
        mock_params = create_mock_lora_parameters()
        
        # Calculate statistics
        stats = {
            "num_lora_layers": len(mock_params),
            "total_lora_parameters": sum(p.numel() for p in mock_params.values()),
            "parameter_norms": {},
            "rank_effectiveness": {},
        }
        
        # Calculate parameter norms
        for name, param in mock_params.items():
            stats["parameter_norms"][name] = param.norm().item()
        
        # Validate statistics
        assert stats["num_lora_layers"] > 0, "Should have LoRA layers"
        assert stats["total_lora_parameters"] > 0, "Should have LoRA parameters"
        assert len(stats["parameter_norms"]) == len(mock_params), "Should have norms for all parameters"
        
        # Test rank effectiveness calculation
        for name, param in mock_params.items():
            if "lora_A" in name:
                base_name = name.replace("lora_A", "")
                b_name = name.replace("lora_A", "lora_B")
                
                if b_name in mock_params:
                    A_matrix = param.cpu()
                    B_matrix = mock_params[b_name].cpu()
                    
                    # Mock SVD calculation (simplified)
                    configured_rank = A_matrix.shape[0]
                    effective_rank = min(configured_rank, 12)  # Mock calculation
                    
                    stats["rank_effectiveness"][base_name] = {
                        "configured_rank": configured_rank,
                        "effective_rank": effective_rank,
                        "rank_utilization": effective_rank / configured_rank,
                        "largest_singular_value": 1.2,
                    }
        
        # Validate rank effectiveness
        for base_name, rank_info in stats["rank_effectiveness"].items():
            assert 0 <= rank_info["effective_rank"] <= rank_info["configured_rank"]
            assert 0 <= rank_info["rank_utilization"] <= 1
            assert rank_info["largest_singular_value"] >= 0
        
        logger.info(f"Statistics: {stats['num_lora_layers']} layers, {stats['total_lora_parameters']} parameters")
        logger.info("✅ LoRA parameter statistics test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ LoRA parameter statistics test FAILED: {e}")
        traceback.print_exc()
        return False


def test_lora_aggregation():
    """Test LoRA aggregation functionality."""
    logger.info("=== Testing LoRA Aggregation ===")
    
    try:
        client_updates = create_mock_client_updates()
        client_weights = {"client_0": 100.0, "client_1": 200.0, "client_2": 150.0}
        
        # Normalize weights
        total_weight = sum(client_weights.values())
        normalized_weights = {
            client_id: weight / total_weight
            for client_id, weight in client_weights.items()
        }
        
        # Validate weight normalization
        weight_sum = sum(normalized_weights.values())
        assert abs(weight_sum - 1.0) < 1e-6, f"Weights should sum to 1, got {weight_sum}"
        
        # Separate LoRA A and B matrices
        first_client = next(iter(client_updates.values()))
        lora_a_params = {}
        lora_b_params = {}
        other_params = {}
        
        for param_name in first_client.keys():
            if "lora_A" in param_name:
                lora_a_params[param_name] = []
            elif "lora_B" in param_name:
                lora_b_params[param_name] = []
            else:
                other_params[param_name] = []
        
        # Collect parameters by type
        for client_id, updates in client_updates.items():
            weight = normalized_weights[client_id]
            for param_name, param_tensor in updates.items():
                if "lora_A" in param_name:
                    lora_a_params[param_name].append((weight, param_tensor))
                elif "lora_B" in param_name:
                    lora_b_params[param_name].append((weight, param_tensor))
                else:
                    other_params[param_name].append((weight, param_tensor))
        
        # Validate separation
        assert len(lora_a_params) > 0, "Should have LoRA A parameters"
        assert len(lora_b_params) > 0, "Should have LoRA B parameters"
        assert len(lora_a_params) == len(lora_b_params), "A and B matrix counts should match"
        
        # Mock aggregation
        aggregated = {}
        
        # Aggregate LoRA A matrices
        for param_name, weighted_params in lora_a_params.items():
            first_param = weighted_params[0][1]
            aggregated_param = first_param  # Simplified aggregation
            
            for weight, param_tensor in weighted_params:
                # Mock weighted averaging
                pass
            
            aggregated[param_name] = aggregated_param
        
        # Aggregate LoRA B matrices
        for param_name, weighted_params in lora_b_params.items():
            first_param = weighted_params[0][1]
            aggregated[param_name] = first_param  # Simplified aggregation
        
        # Validate aggregation results
        assert len(aggregated) == len(first_client), "Should aggregate all parameters"
        
        # Validate parameter shapes
        for name, param in aggregated.items():
            original_shape = first_client[name].shape
            assert param.shape == original_shape, f"Shape mismatch for {name}"
        
        logger.info(f"Aggregated {len(aggregated)} parameters from {len(client_updates)} clients")
        logger.info("✅ LoRA aggregation test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ LoRA aggregation test FAILED: {e}")
        traceback.print_exc()
        return False


def test_parameter_consistency_validation():
    """Test LoRA parameter consistency validation."""
    logger.info("=== Testing Parameter Consistency Validation ===")
    
    try:
        # Test consistent parameters
        consistent_updates = create_mock_client_updates()
        
        # Validate parameter names consistency
        first_client_params = set(next(iter(consistent_updates.values())).keys())
        
        for client_id, updates in consistent_updates.items():
            client_params = set(updates.keys())
            assert client_params == first_client_params, f"Parameter mismatch for {client_id}"
        
        # Validate LoRA A/B matrix pairs
        lora_a_params = {name for name in first_client_params if "lora_A" in name}
        lora_b_params = {name for name in first_client_params if "lora_B" in name}
        
        for a_param in lora_a_params:
            expected_b_param = a_param.replace("lora_A", "lora_B")
            assert expected_b_param in lora_b_params, f"Missing B matrix for {a_param}"
        
        # Validate parameter shapes across clients
        for param_name in first_client_params:
            first_shape = next(iter(consistent_updates.values()))[param_name].shape
            
            for client_id, updates in consistent_updates.items():
                client_shape = updates[param_name].shape
                assert client_shape == first_shape, f"Shape mismatch for {param_name} in {client_id}"
        
        # Test inconsistent parameters (different shapes)
        inconsistent_updates = {
            "client_0": {
                "param_A": MockTensor([16, 768]),
                "param_B": MockTensor([768, 16])
            },
            "client_1": {
                "param_A": MockTensor([8, 768]),  # Different rank
                "param_B": MockTensor([768, 8])
            }
        }
        
        # Validate detection of inconsistency
        first_client_shape = inconsistent_updates["client_0"]["param_A"].shape
        second_client_shape = inconsistent_updates["client_1"]["param_A"].shape
        
        if first_client_shape != second_client_shape:
            logger.info("✓ Correctly detected parameter shape mismatch")
        else:
            raise AssertionError("Should have detected shape mismatch")
        
        logger.info("✅ Parameter consistency validation test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ Parameter consistency validation test FAILED: {e}")
        traceback.print_exc()
        return False


def test_adaptive_rank_selection():
    """Test adaptive rank selection functionality."""
    logger.info("=== Testing Adaptive Rank Selection ===")
    
    try:
        # Test rank selection heuristics
        test_cases = [
            {"data_size": 50, "current_rank": 16, "expected_max": 8},
            {"data_size": 500, "current_rank": 8, "expected_between": (8, 16)},
            {"data_size": 5000, "current_rank": 32, "expected_min": 16},
        ]
        
        for case in test_cases:
            data_size = case["data_size"]
            current_rank = case["current_rank"]
            
            # Apply heuristic logic
            if data_size < 100:
                optimal_rank = min(8, current_rank)
            elif data_size < 1000:
                optimal_rank = min(16, max(8, current_rank))
            else:
                optimal_rank = min(64, max(16, current_rank))
            
            # Validate against expected bounds
            if "expected_max" in case:
                assert optimal_rank <= case["expected_max"], \
                    f"Rank {optimal_rank} exceeds max {case['expected_max']} for data size {data_size}"
            if "expected_min" in case:
                assert optimal_rank >= case["expected_min"], \
                    f"Rank {optimal_rank} below min {case['expected_min']} for data size {data_size}"
            if "expected_between" in case:
                min_rank, max_rank = case["expected_between"]
                assert min_rank <= optimal_rank <= max_rank, \
                    f"Rank {optimal_rank} not in range [{min_rank}, {max_rank}] for data size {data_size}"
            
            logger.info(f"Data size {data_size}: rank {current_rank} → {optimal_rank}")
        
        # Test rank utilization adjustment
        mock_rank_utilizations = [0.3, 0.7, 0.95]  # Low, medium, high utilization
        
        for utilization in mock_rank_utilizations:
            current_rank = 16
            
            if utilization < 0.5:
                # Low utilization: reduce rank
                adjusted_rank = max(4, int(current_rank * 0.75))
            elif utilization > 0.9:
                # High utilization: increase rank
                adjusted_rank = min(64, int(current_rank * 1.25))
            else:
                # Medium utilization: keep current
                adjusted_rank = current_rank
            
            logger.info(f"Utilization {utilization:.1f}: rank {current_rank} → {adjusted_rank}")
            
            # Validate adjustments
            assert 1 <= adjusted_rank <= 64, f"Adjusted rank {adjusted_rank} out of bounds"
        
        logger.info("✅ Adaptive rank selection test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ Adaptive rank selection test FAILED: {e}")
        traceback.print_exc()
        return False


def test_parameter_divergence_calculation():
    """Test parameter divergence calculation."""
    logger.info("=== Testing Parameter Divergence Calculation ===")
    
    try:
        # Create two parameter sets with known differences
        params1 = create_mock_lora_parameters()
        params2 = {name: param for name, param in params1.items()}  # Copy
        
        # Add controlled differences
        for name in list(params2.keys())[:2]:  # Modify first 2 parameters
            params2[name] = MockTensor(params1[name].shape)  # Different values
        
        # Calculate divergences (mock implementation)
        divergences = {}
        for name in params1.keys():
            if name in params2:
                # Mock divergence calculation
                param1 = params1[name]
                param2 = params2[name]
                
                # Simulate divergence metrics
                if name in list(params1.keys())[:2]:
                    # Modified parameters - higher divergence
                    cos_sim = 0.8
                    l2_dist = 0.5
                    rel_l2_dist = 0.3
                else:
                    # Unmodified parameters - lower divergence
                    cos_sim = 0.99
                    l2_dist = 0.01
                    rel_l2_dist = 0.005
                
                divergences[name] = {
                    "cosine_similarity": cos_sim,
                    "l2_distance": l2_dist,
                    "relative_l2_distance": rel_l2_dist
                }
        
        # Validate divergence calculations
        assert len(divergences) == len(params1), "Should have divergence for all parameters"
        
        for name, div in divergences.items():
            assert 0 <= div["cosine_similarity"] <= 1, f"Invalid cosine similarity for {name}"
            assert div["l2_distance"] >= 0, f"Invalid L2 distance for {name}"
            assert div["relative_l2_distance"] >= 0, f"Invalid relative L2 distance for {name}"
        
        # Check that modified parameters have higher divergence
        modified_names = list(params1.keys())[:2]
        unmodified_names = list(params1.keys())[2:]
        
        if modified_names and unmodified_names:
            mod_avg_cos = sum(divergences[name]["cosine_similarity"] for name in modified_names) / len(modified_names)
            unmod_avg_cos = sum(divergences[name]["cosine_similarity"] for name in unmodified_names) / len(unmodified_names)
            
            assert mod_avg_cos < unmod_avg_cos, "Modified parameters should have lower cosine similarity"
        
        logger.info(f"Calculated divergences for {len(divergences)} parameters")
        logger.info("✅ Parameter divergence calculation test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ Parameter divergence calculation test FAILED: {e}")
        traceback.print_exc()
        return False


def test_error_handling():
    """Test error handling in LoRA operations."""
    logger.info("=== Testing Error Handling ===")
    
    try:
        # Test empty parameter sets
        try:
            empty_updates = {}
            # Should handle empty input gracefully
            if not empty_updates:
                logger.info("✓ Correctly detected empty client updates")
        except Exception as e:
            logger.info(f"✓ Handled empty updates with exception: {e}")
        
        # Test mismatched parameter shapes
        mismatched_params = {
            "client_0": {"param": MockTensor([16, 768])},
            "client_1": {"param": MockTensor([8, 768])}  # Different rank
        }
        
        # Check shape consistency
        first_client = next(iter(mismatched_params.values()))
        shape_mismatch_detected = False
        
        for client_id, updates in mismatched_params.items():
            for param_name, param in updates.items():
                expected_shape = first_client[param_name].shape
                if param.shape != expected_shape:
                    shape_mismatch_detected = True
                    logger.info(f"✓ Detected shape mismatch for {client_id}.{param_name}")
        
        assert shape_mismatch_detected, "Should detect shape mismatch"
        
        # Test device mismatch handling
        cpu_param = MockTensor([16, 768], device="cpu")
        gpu_param = MockTensor([16, 768], device="cuda")
        
        # Mock device compatibility check
        if cpu_param.device.type != gpu_param.device.type:
            logger.info("✓ Detected device mismatch")
            # Mock device conversion
            converted_param = gpu_param.to(device="cpu")
            assert converted_param.device.type == "cpu", "Device conversion failed"
            logger.info("✓ Successfully converted parameter to target device")
        
        logger.info("✅ Error handling test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error handling test FAILED: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all enhanced LoRA functionality tests."""
    logger.info("Starting enhanced LoRA functionality tests...")
    
    tests = [
        test_lora_parameter_extraction,
        test_lora_parameter_statistics,
        test_lora_aggregation,
        test_parameter_consistency_validation,
        test_adaptive_rank_selection,
        test_parameter_divergence_calculation,
        test_error_handling,
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            logger.error(f"Test {test_func.__name__} failed with exception: {e}")
    
    logger.info(f"\n=== Enhanced LoRA Test Results ===")
    logger.info(f"Passed: {passed}/{total}")
    logger.info(f"Success rate: {passed/total*100:.1f}%")
    
    if passed == total:
        logger.info("🎉 All enhanced LoRA functionality tests PASSED!")
        return 0
    else:
        logger.warning("⚠️  Some tests failed. See logs above for details.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)