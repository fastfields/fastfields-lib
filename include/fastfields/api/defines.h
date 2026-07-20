#ifndef FF_LIB_DEFINES
#define FF_LIB_DEFINES

#define FF                          ff
#define FF_NAMESPACE_BEGIN(NAME)    namespace NAME {
#define FF_NAMESPACE_END(NAME)      }

#define FF_CPU cpu
#ifndef FF_WITH_CUDA
#  define FF_CUDA notimplemented
#else
#  define FF_CUDA cuda
#endif

#endif // FF_LIB_DEFINES
