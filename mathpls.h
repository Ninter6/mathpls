#pragma once

namespace mathpls {

// useful tool functions

template <class T1, class T2>
constexpr auto max(T1 a, T2 b) {
    return a>b ? a:b;
}

template <class T1, class T2>
constexpr auto min(T1 a, T2 b) {
    return a<b ? a:b;
}

template <class T1, class T2, class T3>
constexpr auto clamp(T1 min, T2 a, T3 max) {
    return (min<(a<max?a:max)?(a<max?a:max):min<(max?a:max)?min:(max<a?a:max));
}

template <class T>
constexpr T abs(T a) {
    return a > 0 ? a : -a;
}

template <class T>
constexpr T pi() {return 3.14159265358979323846264338327950288;}
constexpr float pi() {return 3.14159265358979323846264338327950288;}

template<class T = float>
constexpr T radians(T angle){
    return angle / T{180} * pi<T>();
}

constexpr long double sqrt(long double x) {
    if (x == 1 || x == 0)
        return x;
    double temp = x / 2;
    while (abs(temp - (temp + x / temp) / 2) > 0.000001)
        temp = (temp + x / temp) / 2;
    return temp;
}

constexpr long double pow(long double ori, long double a) {
    if(a < 0) return 1. / pow(ori, -a);
    int ip = a;
    long double fp = a - ip;
    long double r = 1;
    while(ip--) r *= ori;
    constexpr long double c[] = {0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625, 0.001953125, 0.0009765625, 0.00048828125, 0.000244140625, 0.0001220703125, 0.00006103515625, 0.000030517578125, 0.0000152587890625};
    long double t = ori;
    for(int i=0; fp >= c[15]; i++){
        t = sqrt(t);
        if(fp < c[i]) continue;
        fp -= c[i];
        r *= t;
    }
    return r;
}

// 三角函数这里对精度和性能上做了很多取舍,目前基本上已经是最理想的情况了,可以保证小数点后4位没有误差
constexpr long double sin(long double a) {
    if(a < 0) return -sin(-a); // sin(-a) = -sin(a)
    
    constexpr int
    angle[] = {23040, 13601, 7187, 3648, 1831, 916, 458, 229, 115, 57, 29, 14, 7, 4, 2, 1};
    
    long long x = 1000000, y = 0; // x的大小会影响精度,不能太大也不能太小,貌似10^6最好
    long long t = 0, r = a/pi()*180*512;
    while(r > 184320) r -= 184320;
    
    for(int i=0; i<16; i++){
        long long rx = x, ry = y;
        while(t < r){
            rx = x;
            ry = y;
            x = rx - (ry>>i);
            y = ry + (rx>>i);
            t += angle[i];
        }
        if(t == r){
            return (long double)y / sqrt(x*x + y*y);
        }else{
            t -= angle[i];
            x = rx;
            y = ry;
        }
    }
    return (long double)y / sqrt(x*x + y*y);
}

constexpr long double cos(long double a) {
    return sin(pi()/2 - a);
}

constexpr long double tan(long double a) {
    return sin(a) / cos(a);
}

constexpr long double cot(long double a) {
    return cos(a) / sin(a);
}

constexpr long double sec(long double a) {
    return 1 / cos(a);
}

constexpr long double csc(long double a) {
    return 1 / sin(a);
}

constexpr long double atan2(long double y, long double x) {
    constexpr int
    angle[] = {11520, 6801, 3593, 1824, 916, 458, 229, 115, 57, 29, 14, 7, 4, 2, 1};
    
    int x_new{}, y_new{};
    int angleSum = 0;
    
    int lx = x * 1000000;
    int ly = y * 1000000;
    
    for(int i = 0; i < 15; i++)
    {
        if(ly > 0)
        {
            x_new = lx + (ly >> i);
            y_new = ly - (lx >> i);
            lx = x_new;
            ly = y_new;
            angleSum += angle[i];
        }
        else
        {
            x_new = lx - (ly >> i);
            y_new = ly + (lx >> i);
            lx = x_new;
            ly = y_new;
            angleSum -= angle[i];
        }
    }
    return radians<long double>((long double)angleSum / (long double)256);
}

constexpr long double atan(long double a) {
    return atan2(a, 1);
}

constexpr long double acot2(long double x, long double y) {
    return atan2(y, x);
}

constexpr long double acot(long double a) {
    return atan2(1, a);
}

constexpr long double asin2(long double y, long double m) {
    long double x = sqrt(m*m - y*y);
    return atan2(y, x);
}

constexpr long double asin(long double a) {
    return asin2(a, 1);
}

constexpr long double acos2(long double x, long double m) {
    long double y = sqrt(m*m - x*x);
    return atan2(y, x);
}

constexpr long double acos(long double a) {
    return acos2(a, 1);
}

constexpr long double asec2(long double m, long double x) {
    return acos2(x, m);
}

constexpr long double asec(long double a) {
    return asec2(a, 1);
}

constexpr long double acsc2(long double m, long double y) {
    return asin2(y, m);
}

constexpr long double acsc(long double a) {
    return acsc2(a, 1);
}

// structures

#define VEC_MEM_FUNC_IMPL(N) \
auto& operator[](unsigned int n) {return this->asArray[n];} /* non-const */ \
auto operator[](unsigned int n) const {return this->asArray[n];} \
auto value_ptr() {return asArray;} /* non-const */ \
auto value_ptr() const {return asArray;} \
auto operator+() const {return *this;} \
auto operator-() const {return vec<T, N>() - *this;} \
auto& operator+=(T k) const { \
    for (int i=0; i<N; i++) asArray[i] += k;\
    return *this; \
} \
auto operator+(T k) const { \
    auto r = *this; \
    return r += k; \
} \
auto& operator-=(T k) const { \
    for (int i=0; i<N; i++) asArray[i] -= k;\
    return *this; \
} \
auto operator-(T k) const { \
    auto r = *this; \
    return r -= k; \
} \
auto& operator*=(T k) const { \
    for (int i=0; i<N; i++) asArray[i] *= k;\
    return *this; \
} \
auto operator*(T k) const { \
    auto r = *this; \
    return r *= k; \
} \
auto& operator/=(T k) const { \
    for (int i=0; i<N; i++) asArray[i] /= k;\
    return *this; \
} \
auto operator/(T k) const { \
    auto r = *this; \
    return r /= k; \
} \
bool operator!=(vec<T, N> k) const { \
    for (int i=0; i<N; i++) \
        if (asArray[i] != k.asArray[i]) \
            return true; \
    return false; \
} \
bool operator==(vec<T, N> k) const {return !(*this != k);} \
constexpr operator mat<T, 1, N>() const; \
T length_squared() const { \
    T r; \
    for (int i=0; i<N; i++) r += asArray[i]*asArray[i]; \
    return r; \
} \
T length() const {return sqrt(length_squared());} \
auto& normalize() {return *this /= length();} \
auto normalized() const {return *this / length();}

template <class T, unsigned int W, unsigned int H>
struct mat;

template <class T, unsigned int N>
struct vec {
    constexpr operator mat<T, 1, N>() const;
};

template <class T>
struct vec<T, 1> {
    constexpr vec(T x = T{0}) : x{x} {}
    
    union {
        struct { T x; };
        struct { T r; };
        T asArray[1];
    };
    
    VEC_MEM_FUNC_IMPL(1)
};

template <class T>
struct vec<T, 2> {
    constexpr vec(T a = T{0}) : x{a}, y{a} {}
    constexpr vec(T x, T y) : x{x}, y{y} {}
    
    union {
        struct { float x, y; };
        struct { float r, g; };
        T asArray[2];
    };
    
    VEC_MEM_FUNC_IMPL(2)
};

template <class T>
struct vec<T, 3> {
    constexpr vec(T a = T{0}) : x{a}, y{a}, z{a} {}
    constexpr vec(T x, T y, T z) : x{x}, y{y}, z{z} {}
    
    union {
        struct { float x, y, z; };
        struct { float r, g, b; };
        T asArray[3];
    };
    
    VEC_MEM_FUNC_IMPL(3)
};

template <class T>
struct vec<T, 4> {
    constexpr vec(T a = T{0}) : x{a}, y{a}, z{a}, w{a} {}
    constexpr vec(T x, T y, T z, T w) : x{x}, y{y}, z{z}, w{w} {}
    
    union {
        struct { float x, y, z, w; };
        struct { float r, g, b, a; };
        T asArray[4];
    };
    
    VEC_MEM_FUNC_IMPL(4)
};

#undef VEC_MEM_FUNC_IMPL // prevent duplicate code

// normal vec type
using vec1 = vec<float, 1>;
using vec2 = vec<float, 2>;
using vec3 = vec<float, 3>;
using vec4 = vec<float, 4>;

using ivec1 = vec<int, 1>;
using ivec2 = vec<int, 2>;
using ivec3 = vec<int, 3>;
using ivec4 = vec<int, 4>;

using uivec1 = vec<unsigned int, 1>;
using uivec2 = vec<unsigned int, 2>;
using uivec3 = vec<unsigned int, 3>;
using uivec4 = vec<unsigned int, 4>;

using dvec1 = vec<double, 1>;
using dvec2 = vec<double, 2>;
using dvec3 = vec<double, 3>;
using dvec4 = vec<double, 4>;

template <class T, unsigned int W, unsigned int H>
struct mat {
    constexpr mat(T a = T{1}) {
        for (int i = 0; i < min(W, H); i++)
            element[i][i] = a;
    }
    constexpr mat(vec<T, H> e[W]) {
        for (int i = 0; i < W; i++) element[i] = e[i];
    }
    
    template <class...Args, class = decltype(new vec<T, H>[]{Args{}...})>
    constexpr mat(Args...args) {
        static_assert(sizeof...(Args) <= W, "illegal number of parameters");
        const vec<T, H> v[]{args...};
        for (int i = 0; i < sizeof...(Args); i++) element[i] = v[i];
    } // imitation aggregate initialization
    
    vec<T, H> element[W]; // data
    
    auto value_ptr() {return element->value_ptr();} // non-const
    auto value_ptr() const {return element->value_ptr();}
    
    auto& operator[](unsigned int w) {return element[w];} // non-const
    auto operator[](unsigned int w) const {return element[w];}
    
    mat<T, W, H> transposed() const {
        mat<T, W, H> r;
        for(int i=0; i<H; i++)
            for(int j=0; j<W; j++)
                r[j][i] = element[i][j];
        return r;
    }
    
    mat<T, W-1, H-1> cofactor(int x, int y) const {
        mat<T, W-1, H-1> r(0.f);
        for(int i=0, rx=0; i<4; i++, rx++){
            if(i == x) continue;
            for(int j=0, ry=0; j<4; j++){
                if(j == y) continue;
                r[rx][ry++] = element[i][j];
            }
        }
        return r;
    } // 余子式
    
};

// normal mat type
using mat2 = mat<float, 2, 2>;
using mat3 = mat<float, 3, 3>;
using mat4 = mat<float, 4, 4>;

using dmat2 = mat<double, 2, 2>;
using dmat3 = mat<double, 3, 3>;
using dmat4 = mat<double, 4, 4>;

template <class T, unsigned int N>
constexpr vec<T, N>::operator mat<T, 1, N>() const {
    mat<T, 1, N> r{*this};
    return r;
}

template<class T, unsigned int W, unsigned int H, unsigned int M>
constexpr mat<T, W, H> operator*(const mat<T, M, H>& m1, const mat<T, W, M>& m2) {
    mat<T, W, H> r{T{0}};
    for (int i=0; i<H; i++)
        for (int j=0; j<W; j++)
            for (int k=0; k<M; k++)
                r[j][i] += m1[k][i] * m2[j][k];
    return r;
}

// useful funstions

template <class T, unsigned int N>
constexpr T dot(vec<T, N> v1, vec<T, N> v2) {
    T r{0};
    for (int i=0; i<N; i++) r += v1[i] * v2[i];
    return r;
}

template <class T>
constexpr vec<T, 3> cross(vec<T, 3> v1, vec<T, 3> v2){
    mat<T, 3, 3> r(0.f);
    r[2][1]-= r[1][2] = v1.x;
    r[2][0]-= r[0][2]-= v1.y;
    r[1][0]-= r[0][1] = v1.z;
    return r * v2;
}

template <class T, unsigned int N>
constexpr T angle(vec<T, N> v1, vec<T, N> v2){
    return acos(dot(v1, v2) / v1.length() / v2.length());
}

template<class T, int N>
constexpr vec<T, N> reflect(vec<T, N> ori, vec<T, N> normal){
    return ori - 2 * mathpls::dot(ori, normal) * normal;
}

}
