#include <chrono>
template <class Resolution = std::chrono::nanoseconds>
class stopwatch
{
public:
    using clock = std::conditional_t<std::chrono::high_resolution_clock::is_steady,
            std::chrono::high_resolution_clock,
            std::chrono::steady_clock>;

    stopwatch() : timer_(), tic_(), total_time_(0.0) {}
    ~stopwatch() {}

    inline void start() { tic_ = clock::now(); }
    inline void stop() { timer_ = clock::now() - tic_; total_time_ += timer_.count(); }
    inline double time() const { return  timer_.count(); }
    inline double total_time() const { return total_time_; }


private:
    std::chrono::duration<double, typename Resolution::period> timer_;
    typename clock::time_point tic_;
    double total_time_;
};