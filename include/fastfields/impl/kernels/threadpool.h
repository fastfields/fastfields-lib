// Copyright (c) Yasser Asmi
// Released under the MIT License (http://opensource.org/licenses/MIT)
// https://github.com/YasserAsmi/wstpool

#ifndef FF_THREADPOOL_H
#define FF_THREADPOOL_H

#include <atomic>
#include <list>
#include <deque>
#include <future>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <memory>
#include <fastfields/core/defines.h>

FF_NAMESPACE_BEGIN(FF_NS)

class ThreadPool;
std::shared_ptr<ThreadPool> get_global_pool();
size_t set_num_threads(size_t nthreads);
size_t get_num_threads();


class ThreadPool
{
public:
    ThreadPool(size_t count)
    {
        // Setup workers and start their threads
        for (size_t i = 0; i < count; i++)
        {
            mWorkers.emplace_back(this, 1000 + i);
        }
        mStealIter = mWorkers.begin();
        for (auto& wrk : mWorkers)
        {
            wrk.start();
        }
    }
    ~ThreadPool()
    {
        for (auto& wrk : mWorkers)
        {
            wrk.exit();
        }
    }

    template<typename F, typename... ARG>
    auto async(F&& f, ARG&&... args) -> std::future<typename std::result_of<F(ARG...)>::type>
    {
        using rettype = typename std::result_of<F(ARG...)>::type;

        // Create a callwrapper and packaged task, which has a future object and wrap
        // it into a void func with no param using a lambda
        auto task = std::make_shared<std::packaged_task<rettype()>> (
            std::bind(std::forward<F>(f), std::forward<ARG>(args)...));

        std::function<void()> work =
            [task]()
            {
                (*task)();
            };

        // Push work.  Find the queue for this task based on current thread id, to try
        // to keep the new child task on the same worker thread if possible.
        pushWork(work);

        // Return the task future for synchronization
        return task->get_future();
    }

private:
    template <typename T>
    class Queue
    {
    public:
        void push(T& e)
        {
            std::unique_lock<std::mutex> lock(mQMut);
            mQ.push_back(e);
        }
        bool pop(T& e)
        {
            std::unique_lock<std::mutex> lock(mQMut);
            if (mQ.empty())
            {
                return false;
            }
            e = std::move(mQ.front());
            mQ.pop_front();
            return true;
        }
        bool steal(T& e)
        {
            std::unique_lock<std::mutex> lock(mQMut);
            if (mQ.empty())
            {
                return false;
            }
            e = std::move(mQ.back());
            mQ.pop_back();
            return true;
        }
        bool empty()
        {
            std::unique_lock<std::mutex> lock(mQMut);
            return mQ.empty();
        }
    private:
        std::deque<T> mQ;
        std::mutex mQMut;
    };

    class Worker
    {
    public:
        Worker() = delete;
        Worker(ThreadPool* pool, int id) :
            mId(id),
            mPool(pool),
            mExit(false)
        {
        }
        void start()
        {
            mThread = std::thread(&Worker::threadFunc, this);
        }
        void push(std::function<void()>& work)
        {
            mQue.push(work);
            wake();
            mPool->requestSteal();
        }
        // Acquiring mCvMut before notifying is not decoration. The waiter below
        // evaluates its predicate and blocks while holding mCvMut, so taking
        // the same mutex here makes it impossible for a notify to land in
        // between -- which is the classic lost wakeup: the worker checks "no
        // work, not exiting", the pusher enqueues and notifies, and the worker
        // then sleeps on a queue that is no longer empty. With one task and one
        // idle worker that is a hang, not a slowdown.
        void wake()
        {
            std::unique_lock<std::mutex> lock(mCvMut);
            lock.unlock();
            mCv.notify_one();
        }
        bool steal(std::function<void()>& work)
        {
            return mQue.steal(work);
        }
        int id()
        {
            return mId;
        }
        std::thread::id threadId()
        {
            return mThread.get_id();
        }
        void exit()
        {
            mExit = true;
            wake();
            mThread.join();
        }

    private:
        std::thread mThread;
        std::condition_variable mCv;
        std::mutex mCvMut;
        Queue<std::function<void()>> mQue;
        int mId;
        ThreadPool* mPool;
        // Written by whoever calls exit() (the pool's destructor, on the main
        // thread) and read by this worker's own thread, so it must be atomic:
        // as a plain bool the pair is a data race, which is exactly what
        // ThreadSanitizer reports the first time the pool is ever exercised.
        std::atomic<bool> mExit;

        void threadFunc()
        {
            while (!mExit)
            {
                // Try to get work from either this queue or, if none found, steal work from
                // another task queue
                std::function<void()> work;
                bool gotwork = mQue.pop(work);
                if (!gotwork)
                {
                    gotwork = mPool->stealWork(work, mId);
                }

                // If we have work, do the work, else wait.
                if (gotwork)
                {
                    work();
                }
                else
                {
                    // Predicated wait, not a bare one: re-checking the exit
                    // flag and the queue under mCvMut is the other half of the
                    // handshake wake() takes that mutex for, and it also
                    // absorbs spurious wakeups.
                    std::unique_lock<std::mutex> lock(mCvMut);
                    mCv.wait(lock, [this] {
                        return mExit.load() || !mQue.empty();
                    });
                }
            }
        }
    };

    std::list<Worker> mWorkers;
    std::list<Worker>::iterator mStealIter;

    void pushWork(std::function<void()>& work)
    {
        std::thread::id thisid = std::this_thread::get_id();
        for (auto& wrk : mWorkers)
        {
            if (thisid == wrk.threadId())
            {
                wrk.push(work);
                return;
            }
        }
        mWorkers.front().push(work);
    }
    bool stealWork(std::function<void()>& work, int excludeId)
    {
        for (auto& wrk : mWorkers)
        {
            if (wrk.id() != excludeId)
            {
                if (wrk.steal(work))
                {
                    return true;
                }
            }
        }
        return false;
    }
    void requestSteal()
    {
        mStealIter++;
        if (mStealIter == mWorkers.end())
        {
            mStealIter = mWorkers.begin();
        }
        mStealIter->wake();
    }
};

FF_NAMESPACE_END(FF_NS)

#include "threadpool.inl"

#endif // FF_THREADPOOL_H
