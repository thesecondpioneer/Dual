#ifndef _STOPWATCH_H
#define _STOPWATCH_H
#include <chrono>

using namespace std::literals;

template <class Clock = std::chrono::high_resolution_clock>
class stopwatch {
public:
    using clock = Clock;

    stopwatch() : _timer(), _tic(), _total_time(0.0) {}
    ~stopwatch() {}

    inline void start() { _tic = clock::now(); }
    inline void stop() { _timer = clock::now() - _tic; _total_time += _timer.count(); }
    inline double time() { return _timer.count(); }
    inline double total_time() { return _total_time; }

private:
    std::chrono::duration<double> _timer;
    typename clock::time_point _tic;
    double _total_time;
};



#endif
