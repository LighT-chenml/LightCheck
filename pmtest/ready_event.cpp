#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAFunctions.h>
#include <cassert>
#include <mutex>
#include <queue>
#include <unordered_map>

#include "ready_event.h"

struct ReadyEventRegistry {
  std::unordered_map<int, cudaStream_t> cuda_streams;
  std::unordered_map<int, std::queue<cudaEvent_t>> cuda_events;
  std::mutex mutex;
};

static ReadyEventRegistry ready_event_registry;

TorchReadyEvent::TorchReadyEvent() 
{
  device_ = c10::cuda::current_device();

  std::lock_guard<std::mutex> guard(ready_event_registry.mutex);
  auto& queue = ready_event_registry.cuda_events[device_];
  if (!queue.empty()) {
    cuda_event_ = queue.front();
    queue.pop();
  } else {
    C10_CUDA_CHECK(cudaEventCreateWithFlags(
        &cuda_event_, cudaEventBlockingSync | cudaEventDisableTiming));
    //THCudaCheck(cudaEventCreateWithFlags(
    //    &cuda_event_, cudaEventBlockingSync | cudaEventDisableTiming));
  }

  // auto stream = c10::cuda::getCurrentCUDAStream(device_);
  auto& stream = get_extra_stream();
  C10_CUDA_CHECK(cudaEventRecord(cuda_event_, stream));
  //auto stream = THCState_getCurrentStreamOnDevice(state, device_);
  //THCudaCheck(cudaEventRecord(cuda_event_, stream));
}

TorchReadyEvent::~TorchReadyEvent() 
{
  {
    std::lock_guard<std::mutex> guard(ready_event_registry.mutex);
    auto& queue = ready_event_registry.cuda_events[device_];
    queue.push(cuda_event_);
  }
}

bool TorchReadyEvent::Ready() const 
{
  auto status = cudaEventQuery(cuda_event_);
  if (status == cudaErrorNotReady) {
    return false;
  }
  C10_CUDA_CHECK(status);
  //THCudaCheck(status);
  return true;
}

void TorchReadyEvent::RecordEvent()
{
  auto& stream = get_extra_stream();
  C10_CUDA_CHECK(cudaEventRecord(cuda_event_, stream));
}

void create_extra_stream()
{
  int device_ = c10::cuda::current_device();
  auto& extra_stream = ready_event_registry.cuda_streams[device_];
  C10_CUDA_CHECK(cudaStreamCreateWithPriority(
        &extra_stream, cudaStreamNonBlocking, 0));
}

cudaStream_t& get_extra_stream()
{
  int device_ = c10::cuda::current_device();
  return ready_event_registry.cuda_streams[device_];
}

std::shared_ptr<ReadyEvent> RecordReadyEvent() 
{
    return std::make_shared<TorchReadyEvent>();
}