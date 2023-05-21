#ifndef TORCH_COMMON_H
#define TORCH_COMMON_H


enum DeviceType { CPU, GPU };

class ReadyEvent {
public:
  virtual bool Ready() const = 0;
  virtual void RecordEvent() = 0;
  virtual ~ReadyEvent() = default;
};


#endif // TORCH_COMMON_H