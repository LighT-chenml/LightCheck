#ifndef TORCH_READY_EVENT_H
#define TORCH_READY_EVENT_H

#include <cuda_runtime.h>

#include <memory>

#include "common.h"

class TorchReadyEvent : public ReadyEvent {
public:
  TorchReadyEvent();
  ~TorchReadyEvent();
  virtual bool Ready() const override;
  virtual void RecordEvent() override;

private:
  int device_;
  cudaEvent_t cuda_event_ = nullptr;
};


std::shared_ptr<ReadyEvent> RecordReadyEvent();
void create_extra_stream();
cudaStream_t& get_extra_stream();

#endif // TORCH_READY_EVENT_H