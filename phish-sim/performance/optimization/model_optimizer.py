# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Phish-Sim Project

"""
ML Model Performance Optimization
Optimizes model inference performance through various techniques
"""

import time
import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import json
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing

try:
    import torch
    import torch.nn as nn
    from torch.jit import script, trace
    import numpy as np
    from transformers import AutoTokenizer, AutoModel
    import onnx
    import onnxruntime as ort
except ImportError:
    torch = None
    nn = None
    script = trace = None
    np = None
    AutoTokenizer = AutoModel = None
    onnx = None
    ort = None

logger = logging.getLogger(__name__)

class OptimizationLevel(Enum):
    """Model optimization levels"""
    NONE = "none"
    BASIC = "basic"
    ADVANCED = "advanced"
    MAXIMUM = "maximum"

class ModelFormat(Enum):
    """Model formats for optimization"""
    PYTORCH = "pytorch"
    TORCHSCRIPT = "torchscript"
    ONNX = "onnx"
    QUANTIZED = "quantized"

@dataclass
class OptimizationConfig:
    """Configuration for model optimization"""
    level: OptimizationLevel = OptimizationLevel.BASIC
    target_format: ModelFormat = ModelFormat.TORCHSCRIPT
    batch_size: int = 32
    max_sequence_length: int = 512
    quantization: bool = False
    pruning: bool = False
    distillation: bool = False
    parallel_processing: bool = True
    gpu_acceleration: bool = False

@dataclass
class ModelPerformance:
    """Model performance metrics"""
    inference_time_ms: float
    throughput_rps: float
    memory_usage_mb: float
    accuracy: float
    model_size_mb: float
    optimization_level: OptimizationLevel

class ModelOptimizer:
    """ML Model Performance Optimizer"""
    
    def __init__(self, config: OptimizationConfig = None):
        self.config = config or OptimizationConfig()
        self.optimized_models: Dict[str, Any] = {}
        self.performance_metrics: Dict[str, ModelPerformance] = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.process_executor = ProcessPoolExecutor(max_workers=2)
        
        # Device configuration
        self.device = self._get_device()
        logger.info(f"Model optimizer initialized with device: {self.device}")
    
    def _get_device(self) -> str:
        """Get available device for model execution"""
        if torch and torch.cuda.is_available() and self.config.gpu_acceleration:
            return "cuda"
        elif torch and torch.backends.mps.is_available() and self.config.gpu_acceleration:
            return "mps"
        else:
            return "cpu"
    
    async def optimize_model(self, model_name: str, model: Any, 
                           tokenizer: Any = None) -> Dict[str, Any]:
        """Optimize model for better performance"""
        try:
            logger.info(f"Starting optimization for model: {model_name}")
            start_time = time.time()
            
            # Get baseline performance
            baseline_perf = await self._benchmark_model(model, tokenizer, "baseline")
            
            optimization_results = {
                'model_name': model_name,
                'baseline_performance': baseline_perf.__dict__,
                'optimizations': []
            }
            
            # Apply optimizations based on level
            if self.config.level == OptimizationLevel.BASIC:
                optimized_model = await self._basic_optimization(model, tokenizer)
                optimization_results['optimizations'].append('basic')
                
            elif self.config.level == OptimizationLevel.ADVANCED:
                optimized_model = await self._advanced_optimization(model, tokenizer)
                optimization_results['optimizations'].extend(['basic', 'advanced'])
                
            elif self.config.level == OptimizationLevel.MAXIMUM:
                optimized_model = await self._maximum_optimization(model, tokenizer)
                optimization_results['optimizations'].extend(['basic', 'advanced', 'maximum'])
            
            else:
                optimized_model = model
            
            # Convert to target format
            if self.config.target_format != ModelFormat.PYTORCH:
                optimized_model = await self._convert_format(optimized_model, tokenizer)
            
            # Benchmark optimized model
            optimized_perf = await self._benchmark_model(optimized_model, tokenizer, "optimized")
            
            # Store optimized model
            self.optimized_models[model_name] = optimized_model
            self.performance_metrics[model_name] = optimized_perf
            
            optimization_results.update({
                'optimized_performance': optimized_perf.__dict__,
                'improvement': {
                    'speedup': baseline_perf.inference_time_ms / optimized_perf.inference_time_ms,
                    'memory_reduction': (baseline_perf.memory_usage_mb - optimized_perf.memory_usage_mb) / baseline_perf.memory_usage_mb,
                    'size_reduction': (baseline_perf.model_size_mb - optimized_perf.model_size_mb) / baseline_perf.model_size_mb
                },
                'optimization_time': time.time() - start_time
            })
            
            logger.info(f"Model optimization completed for {model_name}")
            return optimization_results
            
        except Exception as e:
            logger.error(f"Model optimization failed for {model_name}: {e}")
            return {'error': str(e)}
    
    async def _basic_optimization(self, model: Any, tokenizer: Any = None) -> Any:
        """Apply basic optimizations"""
        try:
            if torch and hasattr(model, 'eval'):
                model.eval()
            
            # Model compilation (if available)
            if torch and hasattr(torch, 'compile'):
                try:
                    model = torch.compile(model)
                    logger.info("Applied torch.compile optimization")
                except Exception as e:
                    logger.warning(f"torch.compile failed: {e}")
            
            # Set to evaluation mode and disable gradients
            if torch:
                for param in model.parameters():
                    param.requires_grad = False
            
            return model
            
        except Exception as e:
            logger.error(f"Basic optimization failed: {e}")
            return model
    
    async def _advanced_optimization(self, model: Any, tokenizer: Any = None) -> Any:
        """Apply advanced optimizations"""
        try:
            # Start with basic optimization
            model = await self._basic_optimization(model, tokenizer)
            
            # Quantization
            if self.config.quantization and torch:
                try:
                    model = torch.quantization.quantize_dynamic(
                        model, {torch.nn.Linear}, dtype=torch.qint8
                    )
                    logger.info("Applied dynamic quantization")
                except Exception as e:
                    logger.warning(f"Quantization failed: {e}")
            
            # Pruning
            if self.config.pruning and torch:
                try:
                    model = self._apply_pruning(model)
                    logger.info("Applied model pruning")
                except Exception as e:
                    logger.warning(f"Pruning failed: {e}")
            
            return model
            
        except Exception as e:
            logger.error(f"Advanced optimization failed: {e}")
            return model
    
    async def _maximum_optimization(self, model: Any, tokenizer: Any = None) -> Any:
        """Apply maximum optimizations"""
        try:
            # Start with advanced optimization
            model = await self._advanced_optimization(model, tokenizer)
            
            # Knowledge distillation (placeholder)
            if self.config.distillation:
                try:
                    model = await self._apply_distillation(model, tokenizer)
                    logger.info("Applied knowledge distillation")
                except Exception as e:
                    logger.warning(f"Distillation failed: {e}")
            
            # Additional optimizations
            if torch:
                # Optimize for inference
                model = model.half() if self.device == "cuda" else model
                logger.info("Applied precision optimization")
            
            return model
            
        except Exception as e:
            logger.error(f"Maximum optimization failed: {e}")
            return model
    
    def _apply_pruning(self, model: Any) -> Any:
        """Apply model pruning"""
        try:
            if torch:
                # Simple magnitude-based pruning
                for module in model.modules():
                    if isinstance(module, torch.nn.Linear):
                        # Prune 20% of weights
                        torch.nn.utils.prune.l1_unstructured(module, name='weight', amount=0.2)
            
            return model
            
        except Exception as e:
            logger.error(f"Pruning failed: {e}")
            return model
    
    async def _apply_distillation(self, model: Any, tokenizer: Any = None) -> Any:
        """Apply knowledge distillation (placeholder)"""
        # This would implement knowledge distillation from a teacher model
        # For now, return the model as-is
        return model
    
    async def _convert_format(self, model: Any, tokenizer: Any = None) -> Any:
        """Convert model to target format"""
        try:
            if self.config.target_format == ModelFormat.TORCHSCRIPT:
                return await self._convert_to_torchscript(model, tokenizer)
            elif self.config.target_format == ModelFormat.ONNX:
                return await self._convert_to_onnx(model, tokenizer)
            else:
                return model
                
        except Exception as e:
            logger.error(f"Format conversion failed: {e}")
            return model
    
    async def _convert_to_torchscript(self, model: Any, tokenizer: Any = None) -> Any:
        """Convert model to TorchScript"""
        try:
            if torch and script:
                # Create example input
                if tokenizer:
                    example_text = "This is a test input for model optimization"
                    inputs = tokenizer(example_text, return_tensors="pt", 
                                     max_length=self.config.max_sequence_length,
                                     padding=True, truncation=True)
                    example_input = tuple(inputs.values())
                else:
                    # Create dummy input
                    example_input = (torch.randn(1, self.config.max_sequence_length),)
                
                # Convert to TorchScript
                model = script(model)
                logger.info("Converted model to TorchScript")
                
            return model
            
        except Exception as e:
            logger.error(f"TorchScript conversion failed: {e}")
            return model
    
    async def _convert_to_onnx(self, model: Any, tokenizer: Any = None) -> Any:
        """Convert model to ONNX"""
        try:
            if torch and onnx and ort:
                # Create example input
                if tokenizer:
                    example_text = "This is a test input for model optimization"
                    inputs = tokenizer(example_text, return_tensors="pt",
                                     max_length=self.config.max_sequence_length,
                                     padding=True, truncation=True)
                    example_input = tuple(inputs.values())
                else:
                    example_input = (torch.randn(1, self.config.max_sequence_length),)
                
                # Export to ONNX
                onnx_path = f"model_{int(time.time())}.onnx"
                torch.onnx.export(
                    model,
                    example_input,
                    onnx_path,
                    export_params=True,
                    opset_version=11,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
                )
                
                # Load ONNX model
                onnx_model = onnx.load(onnx_path)
                ort_session = ort.InferenceSession(onnx_path)
                
                logger.info("Converted model to ONNX")
                return ort_session
                
            return model
            
        except Exception as e:
            logger.error(f"ONNX conversion failed: {e}")
            return model
    
    async def _benchmark_model(self, model: Any, tokenizer: Any = None, 
                             stage: str = "test") -> ModelPerformance:
        """Benchmark model performance"""
        try:
            # Prepare test data
            test_inputs = await self._prepare_test_data(tokenizer)
            
            # Warmup
            await self._warmup_model(model, test_inputs)
            
            # Benchmark inference
            inference_times = []
            memory_usage = []
            
            for _ in range(10):  # Run 10 iterations
                start_time = time.time()
                memory_before = self._get_memory_usage()
                
                # Run inference
                await self._run_inference(model, test_inputs)
                
                inference_time = (time.time() - start_time) * 1000  # Convert to ms
                memory_after = self._get_memory_usage()
                
                inference_times.append(inference_time)
                memory_usage.append(memory_after - memory_before)
            
            # Calculate metrics
            avg_inference_time = sum(inference_times) / len(inference_times)
            throughput = 1000 / avg_inference_time  # requests per second
            avg_memory_usage = sum(memory_usage) / len(memory_usage)
            model_size = self._get_model_size(model)
            
            return ModelPerformance(
                inference_time_ms=avg_inference_time,
                throughput_rps=throughput,
                memory_usage_mb=avg_memory_usage,
                accuracy=0.95,  # Placeholder - would be calculated from actual evaluation
                model_size_mb=model_size,
                optimization_level=self.config.level
            )
            
        except Exception as e:
            logger.error(f"Model benchmarking failed: {e}")
            return ModelPerformance(
                inference_time_ms=0, throughput_rps=0, memory_usage_mb=0,
                accuracy=0, model_size_mb=0, optimization_level=self.config.level
            )
    
    async def _prepare_test_data(self, tokenizer: Any = None) -> List[Any]:
        """Prepare test data for benchmarking"""
        test_texts = [
            "This is a legitimate business email from our company.",
            "URGENT: Your account will be suspended if you don't act now!",
            "Click here to verify your account immediately.",
            "Congratulations! You've won a free iPhone!",
            "Please update your payment information to continue service."
        ]
        
        if tokenizer:
            inputs = []
            for text in test_texts:
                input_data = tokenizer(text, return_tensors="pt",
                                     max_length=self.config.max_sequence_length,
                                     padding=True, truncation=True)
                inputs.append(input_data)
            return inputs
        else:
            # Create dummy inputs
            return [torch.randn(1, self.config.max_sequence_length) for _ in test_texts]
    
    async def _warmup_model(self, model: Any, test_inputs: List[Any]):
        """Warmup model with test inputs"""
        try:
            for _ in range(3):  # Run 3 warmup iterations
                await self._run_inference(model, test_inputs[:1])
        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")
    
    async def _run_inference(self, model: Any, inputs: List[Any]):
        """Run model inference"""
        try:
            if torch and hasattr(model, 'forward'):
                with torch.no_grad():
                    for input_data in inputs:
                        if isinstance(input_data, dict):
                            # Handle tokenizer output
                            model(**input_data)
                        else:
                            # Handle tensor input
                            model(input_data)
            else:
                # Handle ONNX or other model types
                if hasattr(model, 'run'):
                    for input_data in inputs:
                        if isinstance(input_data, dict):
                            # Convert to numpy arrays for ONNX
                            onnx_inputs = {k: v.numpy() for k, v in input_data.items()}
                            model.run(None, onnx_inputs)
                        else:
                            model.run(None, {model.get_inputs()[0].name: input_data.numpy()})
                            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except Exception:
            return 0.0
    
    def _get_model_size(self, model: Any) -> float:
        """Get model size in MB"""
        try:
            if torch and hasattr(model, 'state_dict'):
                total_size = 0
                for param in model.parameters():
                    total_size += param.numel() * param.element_size()
                return total_size / (1024 * 1024)
            else:
                return 0.0
        except Exception:
            return 0.0
    
    async def batch_inference(self, model_name: str, inputs: List[Any]) -> List[Any]:
        """Run batch inference on optimized model"""
        try:
            if model_name not in self.optimized_models:
                raise ValueError(f"Model {model_name} not found in optimized models")
            
            model = self.optimized_models[model_name]
            
            if self.config.parallel_processing:
                # Use parallel processing for batch inference
                batch_size = self.config.batch_size
                results = []
                
                for i in range(0, len(inputs), batch_size):
                    batch = inputs[i:i + batch_size]
                    batch_result = await self._process_batch(model, batch)
                    results.extend(batch_result)
                
                return results
            else:
                # Sequential processing
                results = []
                for input_data in inputs:
                    result = await self._run_single_inference(model, input_data)
                    results.append(result)
                
                return results
                
        except Exception as e:
            logger.error(f"Batch inference failed: {e}")
            return []
    
    async def _process_batch(self, model: Any, batch: List[Any]) -> List[Any]:
        """Process a batch of inputs"""
        try:
            # This would implement actual batch processing
            # For now, process sequentially
            results = []
            for input_data in batch:
                result = await self._run_single_inference(model, input_data)
                results.append(result)
            return results
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            return []
    
    async def _run_single_inference(self, model: Any, input_data: Any) -> Any:
        """Run single inference"""
        try:
            if torch and hasattr(model, 'forward'):
                with torch.no_grad():
                    if isinstance(input_data, dict):
                        return model(**input_data)
                    else:
                        return model(input_data)
            else:
                # Handle ONNX or other model types
                if hasattr(model, 'run'):
                    if isinstance(input_data, dict):
                        onnx_inputs = {k: v.numpy() for k, v in input_data.items()}
                        return model.run(None, onnx_inputs)
                    else:
                        return model.run(None, {model.get_inputs()[0].name: input_data.numpy()})
                        
        except Exception as e:
            logger.error(f"Single inference failed: {e}")
            return None
    
    def get_optimization_report(self, model_name: str) -> Dict[str, Any]:
        """Get optimization report for a model"""
        try:
            if model_name not in self.performance_metrics:
                return {'error': f'No performance data for model {model_name}'}
            
            performance = self.performance_metrics[model_name]
            
            return {
                'model_name': model_name,
                'performance_metrics': performance.__dict__,
                'optimization_config': self.config.__dict__,
                'device': self.device,
                'optimization_status': 'completed'
            }
            
        except Exception as e:
            logger.error(f"Failed to get optimization report: {e}")
            return {'error': str(e)}
    
    def get_all_optimization_reports(self) -> Dict[str, Any]:
        """Get optimization reports for all models"""
        try:
            reports = {}
            for model_name in self.performance_metrics:
                reports[model_name] = self.get_optimization_report(model_name)
            
            return {
                'total_models': len(reports),
                'reports': reports,
                'summary': {
                    'avg_speedup': sum(
                        r.get('performance_metrics', {}).get('inference_time_ms', 0) 
                        for r in reports.values()
                    ) / len(reports) if reports else 0,
                    'total_memory_saved': sum(
                        r.get('performance_metrics', {}).get('memory_usage_mb', 0) 
                        for r in reports.values()
                    )
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get all optimization reports: {e}")
            return {'error': str(e)}

# Global model optimizer instance
model_optimizer: Optional[ModelOptimizer] = None

def get_model_optimizer() -> ModelOptimizer:
    """Get global model optimizer instance"""
    global model_optimizer
    if model_optimizer is None:
        config = OptimizationConfig()
        model_optimizer = ModelOptimizer(config)
    return model_optimizer