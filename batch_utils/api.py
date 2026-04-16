import json
import math
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Literal
from dataclasses import dataclass, field
from enum import Enum
from openai import OpenAI
from together import Together

from credentials import openai_api_key, together_api_key


class BatchStatus(Enum):
    """Unified batch status across providers."""
    VALIDATING = "validating"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


@dataclass
class BatchRequest:
    """Represents a single request in a batch."""
    custom_id: str
    prompt: str
    model: str
    params: Dict[str, Any] = field(default_factory=dict)
    system_prompt: Optional[str] = None
    response_format: Optional[Dict] = None  # For structured output
    logprobs: bool = False


@dataclass
class BatchResult:
    """Represents a single result from a batch."""
    custom_id: str
    response: Optional[str] = None
    tokens: Optional[List[str]] = None
    logprobs: Optional[List[float]] = None
    error: Optional[str] = None
    raw_response: Optional[Dict] = None


class BatchInferenceBase:
    def __init__(self, cache_dir: str = "./batch_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def create_batch_file(self, requests: List[BatchRequest], batch_name: str) -> Path:
        """Create a JSONL file for batch submission. Override in subclass."""
        raise NotImplementedError
    
    def submit_batch(self, batch_file_path: Path, endpoint: str = "/v1/chat/completions") -> str:
        """Submit a batch job. Returns batch_id. Override in subclass."""
        raise NotImplementedError
    
    def get_batch_status(self, batch_id: str) -> BatchStatus:
        """Get the status of a batch job. Override in subclass."""
        raise NotImplementedError
    
    def get_batch_results(self, batch_id: str) -> List[BatchResult]:
        """Retrieve results from a completed batch. Override in subclass."""
        raise NotImplementedError
    
    def wait_for_batch(self, batch_id: str, poll_interval: int = 60, timeout: int = 86400) -> BatchStatus:
        """Wait for a batch to complete, polling at specified intervals."""
        start_time = time.time()
        while True:
            status = self.get_batch_status(batch_id)
            print(f"Batch {batch_id} status: {status.value}")
            
            if status in [BatchStatus.COMPLETED, BatchStatus.FAILED, BatchStatus.CANCELLED, BatchStatus.EXPIRED]:
                return status
            
            if time.time() - start_time > timeout:
                print(f"Batch {batch_id} timed out after {timeout} seconds")
                return status
            
            time.sleep(poll_interval)
    
    def wait_for_batches(self, batch_ids: List[str], poll_interval: int = 60, timeout: int = 86400) -> Dict[str, BatchStatus]:
        """Wait for multiple batches to complete, polling at specified intervals."""
        terminal_states = {BatchStatus.COMPLETED, BatchStatus.FAILED, BatchStatus.CANCELLED, BatchStatus.EXPIRED}
        statuses = {}
        start_time = time.time()
        
        while True:
            all_done = True
            for bid in batch_ids:
                if bid in statuses and statuses[bid] in terminal_states:
                    continue
                status = self.get_batch_status(bid)
                statuses[bid] = status
                print(f"Batch {bid} status: {status.value}")
                if status not in terminal_states:
                    all_done = False
            
            if all_done:
                return statuses
            
            if time.time() - start_time > timeout:
                print(f"Timed out after {timeout} seconds")
                return statuses
            
            time.sleep(poll_interval)


class OpenAIBatchInference(BatchInferenceBase):
    """Batch inference for OpenAI API."""
    
    def __init__(self, cache_dir: str = "./batch_cache/openai"):
        super().__init__(cache_dir)
        self.client = OpenAI(api_key=openai_api_key)
    
    def create_batch_file(self, requests: List[BatchRequest], batch_name: str) -> Path:
        """Create a JSONL file in OpenAI batch format."""
        file_path = self.cache_dir / f"{batch_name}.jsonl"
        
        with open(file_path, 'w', encoding='utf-8') as f:
            for req in requests:
                messages = []
                if req.system_prompt:
                    messages.append({"role": "system", "content": req.system_prompt})
                messages.append({"role": "user", "content": req.prompt})

                # OpenAI batch API requires max_completion_tokens instead of max_tokens
                if "max_tokens" in req.params:
                    req.params["max_completion_tokens"] = req.params["max_tokens"]
                    del req.params["max_tokens"]
                    del req.params["temperature"]   # OpenAI batch API does not support custom temperature
                
                body = {
                    "model": req.model,
                    "messages": messages,
                    "reasoning_effort": "low",     # Needed for OpenAI's reasoning models for lower token consumption. Also, for UQ tasks we assume no reasoning is needed.
                    **req.params
                }
                
                if req.logprobs:
                    body["logprobs"] = True
                
                if req.response_format:
                    body["response_format"] = req.response_format
                
                line = {
                    "custom_id": req.custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": body
                }
                f.write(json.dumps(line, ensure_ascii=False) + '\n')
        
        return file_path
    
    def submit_batch(self, batch_file_path: Path, endpoint: str = "/v1/chat/completions") -> str:
        """Submit a batch job to OpenAI."""
        # Upload the file
        with open(batch_file_path, 'rb') as f:
            file_response = self.client.files.create(file=f, purpose="batch")
        
        print(f"Uploaded file: {file_response.id}")
        
        # Create the batch
        batch = self.client.batches.create(
            input_file_id=file_response.id,
            endpoint=endpoint,
            completion_window="24h"
        )
        
        print(f"Created batch: {batch.id}")
        
        # Save batch info
        batch_info_path = self.cache_dir / f"{batch.id}_info.json"
        with open(batch_info_path, 'w') as f:
            json.dump({
                "batch_id": batch.id,
                "input_file_id": file_response.id,
                "batch_file_path": str(batch_file_path),
                "created_at": time.time()
            }, f, indent=2)
        
        return batch.id
    
    def get_batch_status(self, batch_id: str) -> BatchStatus:
        """Get the status of an OpenAI batch."""
        batch = self.client.batches.retrieve(batch_id)
        
        status_map = {
            "validating": BatchStatus.VALIDATING,
            "in_progress": BatchStatus.IN_PROGRESS,
            "finalizing": BatchStatus.IN_PROGRESS,
            "completed": BatchStatus.COMPLETED,
            "failed": BatchStatus.FAILED,
            "cancelled": BatchStatus.CANCELLED,
            "cancelling": BatchStatus.CANCELLED,
            "expired": BatchStatus.EXPIRED
        }
        
        return status_map.get(batch.status, BatchStatus.IN_PROGRESS)
    
    def get_batch_info(self, batch_id: str) -> Dict:
        """Get detailed batch information."""
        batch = self.client.batches.retrieve(batch_id)
        return {
            "id": batch.id,
            "status": batch.status,
            "output_file_id": batch.output_file_id,
            "error_file_id": batch.error_file_id,
            "request_counts": {
                "total": batch.request_counts.total,
                "completed": batch.request_counts.completed,
                "failed": batch.request_counts.failed
            } if batch.request_counts else None
        }
    
    def get_batch_results(self, batch_id: str) -> List[BatchResult]:
        """Retrieve results from a completed OpenAI batch."""
        batch = self.client.batches.retrieve(batch_id)
        
        if not batch.output_file_id:
            print(f"No output file available for batch {batch_id}")
            return []
        
        # Download the output file
        file_response = self.client.files.content(batch.output_file_id)
        content = file_response.text
        
        # Save raw output
        output_path = self.cache_dir / f"{batch_id}_output.jsonl"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # Parse results
        results = []
        for line in content.strip().split('\n'):
            if not line:
                continue
            data = json.loads(line)
            
            result = BatchResult(custom_id=data["custom_id"])
            
            if data.get("error"):
                result.error = data["error"].get("message", str(data["error"]))
            elif data.get("response") and data["response"].get("body"):
                body = data["response"]["body"]
                result.raw_response = body
                
                if body.get("choices"):
                    choice = body["choices"][0]
                    result.response = choice["message"]["content"].strip()
                    
                    # Extract logprobs if available
                    if choice.get("logprobs") and choice["logprobs"].get("content"):
                        result.tokens = [c["token"] for c in choice["logprobs"]["content"]]
                        result.logprobs = [c["logprob"] for c in choice["logprobs"]["content"]]
            
            results.append(result)
        
        return results
    
    def list_batches(self, limit: int = 10) -> List[Dict]:
        """List recent batches."""
        batches = self.client.batches.list(limit=limit)
        return [{"id": b.id, "status": b.status, "created_at": b.created_at} for b in batches]


class TogetherAIBatchInference(BatchInferenceBase):
    """Batch inference for TogetherAI API."""
    
    def __init__(self, cache_dir: str = "./batch_cache/togetherai"):
        super().__init__(cache_dir)
        self.client = Together(api_key=together_api_key)
    
    def create_batch_file(self, requests: List[BatchRequest], batch_name: str) -> Path:
        """Create a JSONL file in TogetherAI batch format."""
        file_path = self.cache_dir / f"{batch_name}.jsonl"
        
        with open(file_path, 'w', encoding='utf-8') as f:
            for req in requests:
                messages = []
                if req.system_prompt:
                    messages.append({"role": "system", "content": req.system_prompt})
                messages.append({"role": "user", "content": req.prompt})
                
                body = {
                    "model": req.model,
                    "messages": messages,
                    **req.params
                }
                
                if req.logprobs:
                    body["logprobs"] = True
                
                if req.response_format:
                    body["response_format"] = req.response_format
                
                # TogetherAI format doesn't include method/url
                line = {
                    "custom_id": req.custom_id,
                    "body": body
                }
                f.write(json.dumps(line, ensure_ascii=False) + '\n')
        
        return file_path
    
    def submit_batch(self, batch_file_path: Path, endpoint: str = "/v1/chat/completions") -> str:
        """Submit a batch job to TogetherAI."""
        # Upload the file
        file_response = self.client.files.upload(
            file=str(batch_file_path),
            purpose="batch-api",
            check=False
        )
        
        print(f"Uploaded file: {file_response.id}")
        
        # Create the batch
        batch = self.client.batches.create(
            input_file_id=file_response.id,
            endpoint=endpoint
        )
        
        batch_id = batch.id if hasattr(batch, 'id') else batch.job.id
        print(f"Created batch: {batch_id}")
        
        # Save batch info
        batch_info_path = self.cache_dir / f"{batch_id}_info.json"
        with open(batch_info_path, 'w') as f:
            json.dump({
                "batch_id": batch_id,
                "input_file_id": file_response.id,
                "batch_file_path": str(batch_file_path),
                "created_at": time.time()
            }, f, indent=2)
        
        return batch_id
    
    def get_batch_status(self, batch_id: str) -> BatchStatus:
        """Get the status of a TogetherAI batch."""
        batch = self.client.batches.retrieve(batch_id)
        
        status_map = {
            "VALIDATING": BatchStatus.VALIDATING,
            "IN_PROGRESS": BatchStatus.IN_PROGRESS,
            "COMPLETED": BatchStatus.COMPLETED,
            "FAILED": BatchStatus.FAILED,
            "CANCELLED": BatchStatus.CANCELLED
        }
        
        return status_map.get(batch.status, BatchStatus.IN_PROGRESS)
    
    def get_batch_info(self, batch_id: str) -> Dict:
        """Get detailed batch information."""
        batch = self.client.batches.retrieve(batch_id)
        return {
            "id": batch.id,
            "status": batch.status,
            "output_file_id": getattr(batch, 'output_file_id', None),
            "error_file_id": getattr(batch, 'error_file_id', None),
            "request_count": getattr(batch, 'request_count', None)
        }
    
    def get_batch_results(self, batch_id: str) -> List[BatchResult]:
        """Retrieve results from a completed TogetherAI batch."""
        batch = self.client.batches.retrieve(batch_id)
        
        if not batch.output_file_id:
            print(f"No output file available for batch {batch_id}")
            return []
        
        # Download the output file
        output_path = self.cache_dir / f"{batch_id}_output.jsonl"

        # self.client.files.content(
        #     id=batch.output_file_id,
        #     output=str(output_path)
        # )
        output = self.client.files.content(
            id=batch.output_file_id,
        )
        # output = output.read()
        # from rich import print as rprint
        # import sys
        # rprint(output.read())
        # sys.exit()
        with open(output_path, 'wb') as f:
            f.write(output.read())
        
        # Parse results
        results = []
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                
                result = BatchResult(custom_id=data["custom_id"])
                
                if data.get("error"):
                    result.error = data["error"].get("message", str(data["error"]))
                elif data.get("response") and data["response"].get("body"):
                    body = data["response"]["body"]
                    result.raw_response = body
                    
                    if body.get("choices"):
                        choice = body["choices"][0]
                        result.response = choice["message"]["content"].strip()
                        
                        # Extract logprobs if available
                        if choice.get("logprobs"):
                            result.tokens = choice["logprobs"].get("tokens", [])
                            result.logprobs = choice["logprobs"].get("token_logprobs", [])
                
                results.append(result)
        
        return results
    
    def list_batches(self, limit: int = 10) -> List[Dict]:
        """List recent batches."""
        batches = self.client.batches.list()
        return [{"id": b.id, "status": b.status} for b in list(batches)[:limit]]


def get_batch_inference(provider: Literal["openai", "togetherai"], **kwargs) -> BatchInferenceBase:
    """Factory function to get the appropriate batch inference handler."""
    if provider == "openai":
        return OpenAIBatchInference(**kwargs)
    elif provider == "togetherai":
        return TogetherAIBatchInference(**kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}")


class BatchRequestCollector:
    """
    Helper class to collect batch requests and manage their submission.
    
    This class helps collect requests during a pipeline run, then submit them
    as a batch and map results back to the original requests.
    """
    
    def __init__(
        self, 
        model_name: str, 
        provider: Literal["openai", "togetherai"], 
        cache_dir: str = "./batch_cache", 
        prefix: str = "req", 
        max_batch_size: int = 10000
    ):
        self.model_name = model_name
        self.provider = provider
        self.requests: List[BatchRequest] = []
        self.request_map: Dict[str, Dict[str, Any]] = {}  # Maps custom_id to metadata
        self.batch_handler = get_batch_inference(provider, cache_dir=f"{cache_dir}/{provider}")
        self._counter = 0
        self.max_batch_size = max_batch_size
    
    def add_request(
        self,
        prompt: str,
        params: Dict[str, Any] = None,
        system_prompt: str = None,
        response_format: Dict = None,
        logprobs: bool = False,
        metadata: Dict[str, Any] = None,
        custom_id_prefix: str = None
    ) -> str:
        """
        Add a request to the batch.
        Returns the custom_id for later retrieval.
        """
        custom_id = f"{custom_id_prefix}_{self._counter}"
        self._counter += 1
        
        request = BatchRequest(
            custom_id=custom_id,
            prompt=prompt,
            model=self.model_name,
            params=params or {},
            system_prompt=system_prompt,
            response_format=response_format,
            logprobs=logprobs
        )
        
        self.requests.append(request)
        self.request_map[custom_id] = {
            "prompt": prompt,
            "metadata": metadata or {}
        }
        
        return custom_id
    
    def submit_batch(self, batch_name: str, endpoint: str = "/v1/chat/completions") -> Union[str, List[str]]:
        """Submit all collected requests as a batch.
        
        If requests exceed max_batch_size, splits into multiple batches.
        Returns a single batch_id (str) or a list of batch_ids (List[str]).
        """
        if not self.requests:
            raise ValueError("No requests to submit")
        
        # Save request map for later
        map_file = self.batch_handler.cache_dir / f"{batch_name}_request_map.json"
        with open(map_file, 'w', encoding='utf-8') as f:
            json.dump(self.request_map, f, indent=2, ensure_ascii=False)
        
        # Split into multiple batches if needed
        if len(self.requests) > self.max_batch_size:
            return self._submit_split_batches(batch_name, endpoint)
        
        # Single batch
        batch_file = self.batch_handler.create_batch_file(self.requests, batch_name)
        print(f"Created batch file with {len(self.requests)} requests: {batch_file}")
        
        batch_id = self.batch_handler.submit_batch(batch_file, endpoint)
        
        info_file = self.batch_handler.cache_dir / f"{batch_name}_batch_info.json"
        with open(info_file, 'w') as f:
            json.dump({
                "batch_id": batch_id,
                "batch_name": batch_name,
                "num_requests": len(self.requests)
            }, f, indent=2)
        
        return batch_id
    
    def _submit_split_batches(self, batch_name: str, endpoint: str) -> List[str]:
        """Split requests into chunks and submit each as a separate batch."""
        num_parts = math.ceil(len(self.requests) / self.max_batch_size)
        print(f"\nSplitting {len(self.requests)} requests into {num_parts} batches "
              f"(max {self.max_batch_size} per batch)")
        
        batch_ids = []
        for i in range(num_parts):
            start = i * self.max_batch_size
            end = min((i + 1) * self.max_batch_size, len(self.requests))
            chunk = self.requests[start:end]
            
            part_name = f"{batch_name}_part{i+1}of{num_parts}"
            batch_file = self.batch_handler.create_batch_file(chunk, part_name)
            print(f"  Part {i+1}/{num_parts}: {len(chunk)} requests -> {batch_file}")
            
            batch_id = self.batch_handler.submit_batch(batch_file, endpoint)
            batch_ids.append(batch_id)
            print(f"  Submitted part {i+1}/{num_parts}: {batch_id}")
        
        # Save group info
        info_file = self.batch_handler.cache_dir / f"{batch_name}_batch_info.json"
        with open(info_file, 'w') as f:
            json.dump({
                "batch_ids": batch_ids,
                "batch_name": batch_name,
                "num_requests": len(self.requests),
                "num_parts": num_parts,
                "max_batch_size": self.max_batch_size
            }, f, indent=2)
        
        print(f"\nAll {num_parts} batch parts submitted: {batch_ids}")
        return batch_ids
    
    def get_status(self, batch_id: Union[str, List[str]]) -> Union[BatchStatus, Dict[str, BatchStatus]]:
        """Get batch status. Accepts a single batch_id or a list of batch_ids."""
        if isinstance(batch_id, list):
            return {bid: self.batch_handler.get_batch_status(bid) for bid in batch_id}
        return self.batch_handler.get_batch_status(batch_id)
    
    def get_results(self, batch_id: Union[str, List[str]]) -> Dict[str, BatchResult]:
        """
        Get batch results mapped by custom_id.
        Accepts a single batch_id or a list of batch_ids; merges results.
        """
        batch_ids = batch_id if isinstance(batch_id, list) else [batch_id]
        all_results = {}
        for bid in batch_ids:
            results = self.batch_handler.get_batch_results(bid)
            all_results.update({r.custom_id: r for r in results})
        return all_results
    
    def clear(self):
        """Clear collected requests."""
        self.requests = []
        self.request_map = {}
        self._counter = 0
    
    @property
    def num_requests(self) -> int:
        """Number of collected requests."""
        return len(self.requests)


def load_batch_results(batch_id: Union[str, List[str]], provider: str, cache_dir: str = "./batch_cache") -> Dict[str, BatchResult]:
    """
    Utility function to load results from a completed batch.
    Accepts a single batch_id or a list of batch_ids; merges results.
    """
    handler = get_batch_inference(provider, cache_dir=f"{cache_dir}/{provider}")
    batch_ids = batch_id if isinstance(batch_id, list) else [batch_id]
    all_results = {}
    for bid in batch_ids:
        results = handler.get_batch_results(bid)
        all_results.update({r.custom_id: r for r in results})
    return all_results


def check_batch_status(batch_id: Union[str, List[str]], provider: str, cache_dir: str = "./batch_cache") -> Union[Dict, List[Dict]]:
    """
    Utility function to check batch status with detailed info.
    Accepts a single batch_id or a list of batch_ids.
    """
    handler = get_batch_inference(provider, cache_dir=f"{cache_dir}/{provider}")
    
    if isinstance(batch_id, list):
        return [
            {"batch_id": bid, "status": handler.get_batch_status(bid).value, "details": handler.get_batch_info(bid)}
            for bid in batch_id
        ]
    
    status = handler.get_batch_status(batch_id)
    info = handler.get_batch_info(batch_id)
    return {
        "status": status.value,
        "details": info
    }
