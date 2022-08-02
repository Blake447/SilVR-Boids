
// Transcribed library from GLSL

#ifndef ROOT_HALF
    #define ROOT_HALF .707106781
#endif

#ifndef PI
    #define PI 3.14159265
#endif

float sgn(float x)
{
    return step(0, x) * 2 - 1;
}

float2 sgn(float2 v)
{
    return float2(step(0, v.x) * 2 - 1, step(0, v.y) * 2 - 1);
}

float lengthSqr(float3 x)
{
    return dot(x, x);
}

float vmax(float2 v)
{
    return max(v.x, v.y);
}

float vmax(float3 v)
{
    return max(max(v.x, v.y), v.z);
}

float vmax(float4 v)
{
    return max(max(max(v.x, v.y), v.z), v.w);
}

float vmin(float2 v)
{
    return min(v.x, v.y);
}

float vmin(float3 v)
{
    return min(min(v.x, v.y), v.z);
}

float vmin(float4 v)
{
    return min(min(min(v.x, v.y), v.z), v.w);
}

void pR(inout float2 p, float a)
{
    p = cos(a) * p + sin(a) * float2(p.y, -p.x);
}

void pR45(inout float2 p)
{
    p = (p + float2(p.y, -p.x)) * ROOT_HALF;
}

void pR45(inout float2 p, float2 offset)
{
    p = (p - offset + float2(p.y - offset.y, -p.x + offset.x)) * ROOT_HALF + offset;
}

void pMod1(inout float p, float size)
{
    p = (frac(p / size + 0.5) - 0.5) * size;
				//return frac(p / size) * size;
				//float halfsize = size * 0.5;
				//float c = floor((p + halfsize) / size);
				//p = fmod(abs(p) + halfsize, size) - halfsize;
				//return c;
}
float modSize(float p, float size)
{
    return (frac((p / size) + 0.5) - 0.5) * size;
}


float pModSingle1(inout float p, float size)
{
    float halfsize = size * .5;
    float c = floor((p + halfsize) / size);
    if (p >= 0)
        p = fmod(p + halfsize, size) - halfsize;
    return c;
}

float pModPolar(inout float2 p, float repetitions)
{
    float angle = 2 * PI / repetitions;
    float r = length(p);
    float a = (atan2(p.x, p.y) + PI) + angle * 0.5;
    float c = floor(a / angle);
    a = fmod(a, angle) - angle * 0.5;
    p = float2(cos(a), sin(a)) * r;
    if (abs(c) >= (repetitions / 2))
        c = abs(c);
    return c;
}

float pModPolar(inout float2 p, float repetitions, float phase)
{
    float angle = 2 * PI / repetitions;
    float r = length(p);
    float a = (atan2(p.x, p.y) + PI + (2 * PI * phase / 360)) + angle * 0.5;
    float c = floor(a / angle);
    a = fmod(a, angle) - angle * 0.5;
    p = float2(cos(a), sin(a)) * r;
    if (abs(c) >= (repetitions / 2))
        c = abs(c);
    return c;
}

void pModMirror(inout float p)
{
    p = abs(p);
}

float fOpUnionChamfer(float a, float b, float r)
{
    return min(min(a, b), (a - r + b) * ROOT_HALF);
}

float fOpIntersectionChamfer(float a, float b, float r)
{
    return max(max(a, b), (a - r + b) * ROOT_HALF);
}

float fOpDifferenceChamfer(float a, float b, float r)
{
    return fOpIntersectionChamfer(a, -b, r);
}

float fOpUnionRound(float a, float b, float r)
{
    float2 u = max(float2(r + a, r + b), float2(0.0, 0.0));
    return max(r, min(a, b)) - length(u);
}

float fOpIntersectionRound(float a, float b, float r)
{
    float2 u = max(float2(r + a, r + b), float2(0.0, 0.0));
    return min(r, min(a, b)) + length(u);
}

float fOpDifferenceRound(float a, float b, float r)
{
    return fOpIntersectionRound(a, -b, r);
}

float fOpUnionSoft(float a, float b, float r)
{
    float e = max(r - abs(a - b), 0);
    return min(a, b) - e * e * 0.25 / r;
}
			
float fOpUnionStairs(float a, float b, float r, float n)
{
    float s = r / n;
    float u = b - r;
    return min(min(a, b), 0.5 * (u + a + abs((fmod(u - a + s, 2 * s)) - s)));
}

float fOpDifferenceStairs(float a, float b, float r, float n)
{
    return -fOpUnionStairs(-a, b, r, n);
}

float fOpPipe(float a, float b, float r)
{
    return length(float2(a, b)) - r;
}

float fOpGroove(float a, float b, float ra, float rb)
{
    return max(a, min(a + ra, rb - abs(b)));
}

float fOpTongue(float a, float b, float ra, float rb)
{
    return min(a, max(a - ra, abs(b) - rb));
}

float fPlane(float3 p, float3 n, float3 c)
{
    float dist = dot(p, n) + c;
    return dist;
}

float fSphere(float3 p, float r, float3 c)
{
    return length(p - c) - r;
}
			
float fBox(float3 p, float3 b)
{
    float3 d = abs(p) - b;
    return max(max(d.x, d.y), d.z);
}

float fBox(float3 p, float3 b, float3 c)
{
    float3 d = abs(p - c) - b;
    return max(max(d.x, d.y), d.z);
}

float fPillar(float3 p, float3 c, float r, float height)
{
    float d = length(p.xz - c.xz) - r;
    d = max(d, abs(p.y - c.y) - height);
    return d;
}

float sdTorus(float3 p, float2 t)
{
    float2 q = float2(length(p.xz) - t.x, p.y);
    return length(q) - t.y;
}

void pExtrude(inout float3 p, float3 h)
{
    p = p - clamp(p, -h, h);
}

float fCapsule(float3 p, float h, float r, float3 c)
{
    float3 q = p - c;
    q.y -= clamp(q.y, 0.0, h);
    return length(q) - r;
}

float smin(float a, float b, float k)
{
    float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    return lerp(b, a, h) - k * h * (1.0 - h);
}

