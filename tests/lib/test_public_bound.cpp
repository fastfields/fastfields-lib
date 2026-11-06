// Standalone unit test for the public-API bound guard, ff::require_public_bound
// (api/checks.h).
//
// `bound::type::NoCheck` disables bounds checking: the kernels *assume* every
// coordinate is in the field of view, so an out-of-range one is an
// out-of-bounds memory access. The C++ layers keep it as an internal fast
// path; the hub (`ff::<fn>(DLTensor&, ...)`) refuses it. Three things are
// checked here, none of which needs a library link:
//
//   1. the guard throws std::invalid_argument, with the documented message, on
//      NoCheck and on nothing else;
//   2. the literal it compares against is the same value as ff::bound_t::NoCheck
//      (the hub's spelling, shared by the api headers), so the two cannot
//      drift. (The kernels' `bound::type` enum cannot be included alongside
//      it -- both define `ff::bound_t` -- and is tied to the same int8 values
//      by the hub->backend casts themselves.)
//   3. every hub entry point in src/lib/*.cpp that takes an `int8_t bound`
//      calls require_public_bound(bound) as its FIRST statement, before any
//      device check or dispatch. This one reads the sources, so that adding
//      an op without the guard fails here rather than shipping a public
//      path to NoCheck: there is no GPU and no hub link in this suite, so a
//      behavioural test per entry point is not available.
//
// Build it via `make test-lib`, or directly from the repo root:
//     clang++ -std=c++11 -Iinclude tests/lib/test_public_bound.cpp -o /tmp/t && /tmp/t

#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>
#include <dirent.h>
#include <fastfields/api/checks.h>
#include <fastfields/api/resize.h>   // ff::bound_t (shared by the api headers)

static int failures = 0;

#define CHECK(cond, msg)                                                       \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::printf("FAIL: %s\n", msg);                                    \
            ++failures;                                                        \
        }                                                                      \
        else {                                                                 \
            std::printf("ok:   %s\n", msg);                                    \
        }                                                                      \
    } while (0)

static std::string threw_with(int8_t bound)
{
    try {
        ff::require_public_bound(bound);
    }
    catch (const std::invalid_argument & e) {
        return std::string(e.what());
    }
    return std::string();
}

// The repository root, from this file's own path (the Makefile compiles it by
// absolute path). FF_TEST_ROOTDIR overrides, for a build from elsewhere.
static std::string root_dir()
{
#ifdef FF_TEST_ROOTDIR
    return FF_TEST_ROOTDIR;
#else
    std::string f = __FILE__;
    const std::string tail = "tests/lib/test_public_bound.cpp";
    if (f.size() >= tail.size() &&
        f.compare(f.size() - tail.size(), tail.size(), tail) == 0)
        return f.substr(0, f.size() - tail.size());
    return "./";
#endif
}

static bool is_ident(char c)
{
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
           (c >= '0' && c <= '9') || c == '_';
}

// Number of `int8_t <ws> bound` parameter declarations in a definition (not a
// prototype) whose body does not open with `require_public_bound(bound);`.
// `seen` counts every such definition, guarded or not.
static int unguarded(const std::string & src, const std::string & name,
                     int & seen)
{
    int bad = 0;
    size_t pos = 0;
    while ((pos = src.find("int8_t", pos)) != std::string::npos) {
        size_t q = pos + 6;
        pos = q;
        if (pos > 6 && is_ident(src[pos - 7])) continue;
        size_t ws = q;
        while (q < src.size() && (src[q] == ' ' || src[q] == '\t')) ++q;
        if (q == ws) continue;
        if (src.compare(q, 5, "bound") != 0) continue;
        if (q + 5 < src.size() && is_ident(src[q + 5])) continue;
        // A definition opens a body before it ends a statement.
        size_t brace = src.find('{', q), semi = src.find(';', q);
        if (brace == std::string::npos || semi < brace) continue;
        ++seen;
        size_t b = brace + 1;
        while (b < src.size() && std::strchr(" \t\r\n", src[b])) ++b;
        if (src.compare(b, 28, "require_public_bound(bound);") != 0) {
            // Report the function name: the last identifier before '('.
            size_t paren = src.rfind('(', q);
            size_t e = paren;
            while (e > 0 && std::strchr(" \t", src[e - 1])) --e;
            size_t s = e;
            while (s > 0 && is_ident(src[s - 1])) --s;
            std::printf("  unguarded: %s: %s\n", name.c_str(),
                        src.substr(s, e - s).c_str());
            ++bad;
        }
    }
    return bad;
}

int main()
{
    // 1. NoCheck throws, with the documented message.
    {
        std::string msg = threw_with(ff::bound_t::NoCheck);
        CHECK(msg == "fastfields: bound NoCheck is internal-only and not "
                     "accepted by the public API",
              "bound=nocheck throws with the documented message");
    }

    // 1b. Nothing else throws: the seven real conditions, Dynamic, and
    //     out-of-range values (left to the backend's own error).
    {
        bool any = false;
        for (int b = -128; b <= 127; ++b)
            if (b != ff::bound_t::NoCheck && !threw_with((int8_t)b).empty())
                any = true;
        CHECK(!any, "every other int8 bound value is left alone");
    }

    // 2. The two spellings of NoCheck agree.
    CHECK(ff::internal_bound_nocheck == ff::bound_t::NoCheck,
          "checks.h literal == ff::bound_t::NoCheck");

    // 3. Every hub entry point taking a bound calls the guard first.
    {
        const std::string dir = root_dir() + "src/lib/";
        std::vector<std::string> files;
        if (DIR * d = opendir(dir.c_str())) {
            while (dirent * e = readdir(d)) {
                std::string n = e->d_name;
                if (n.size() > 4 && n.compare(n.size() - 4, 4, ".cpp") == 0)
                    files.push_back(n);
            }
            closedir(d);
        }
        std::sort(files.begin(), files.end());
        CHECK(!files.empty(), "found the hub sources (src/lib/*.cpp)");

        int seen = 0, bad = 0;
        for (size_t i = 0; i < files.size(); ++i) {
            std::ifstream in((dir + files[i]).c_str());
            std::stringstream ss;
            ss << in.rdbuf();
            bad += unguarded(ss.str(), files[i], seen);
        }
        std::printf("  %d hub entry point(s) take a bound\n", seen);
        CHECK(seen > 0, "found hub entry points taking a bound");
        CHECK(bad == 0,
              "every one of them calls require_public_bound(bound) first");
    }

    if (failures) {
        std::printf("\n%d check(s) FAILED\n", failures);
        return 1;
    }
    std::printf("\nAll checks passed.\n");
    return 0;
}
