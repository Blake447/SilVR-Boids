Shader "SilVR/Boids/Simulator 32x32-testBoids-main    "
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
        _VelocityScale("Velocity Scale", float) = 1.175
        _TimeScale("Time Scale", float) = 0.015

        _RepulsionScale("Repulsion Scale", float) = 150
        _RepulsionInfluence("Repulsion Influence", Range(.0001, .25)) = .0001
        _RepulsionDistance("Repulsion Distance", float) = 0.1

        _AlignmentScale("Alignment Scale", float) = 0.25
        _AlignmentInfluence("Alignment Influence", Range(.0001, .25)) = .0001

        _GroupingScale("Grouping Scale", float) = 0.45
        _GroupingInfluence("Grouping Influence", Range(.0001, .25)) = .0001

        _BoundaryScale("Boundary Scale", float) = 35
        _BoundaryInfluence("Boundary Influence", Range(0.01, .25)) = 0.05
        _BoundaryDistance("Boundary Distance", float) = 1
        _BoundaryMultipliers("Boundary Multipliers", Vector) = (1,1,1,1)

        _ClosenessScale("Closeness Scale", float) = 25
        _ExclusionScale("Exclusion Scale", float) = 0.0001
        _SpeedLimit("Speed limit scale", float) = .00005
        _WorldPosTarget("Target Texture", 2D) = "black" {}
        _Noise("Noise Texture", 2D) = "black" {}
        _Damping("Velocity Damping", Range(0.9,1.0)) = 0.9
        _SpeedNosie("Speed Nosie", float) = 0.1

        _TextureWidth("Texture width", int) = 16
        _TextureHeight("Texture height", int) = 16
        _BoidCount("Boid count", int) = 256

       

    }
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        LOD 100

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            // make fog work
            #pragma multi_compile_fog

            #include "UnityCG.cginc"

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float2 uv : TEXCOORD0;
                UNITY_FOG_COORDS(1)
                float4 vertex : SV_POSITION;
            };

            struct ParticleData
            {
                float4 pos;
                float4 vel;
            };

            sampler2D _MainTex;
            sampler2D _WorldPosTarget;

            float4 _MainTex_ST;
            float _VelocityScale;

            float _TimeScale;
            
            float _AttractionScale;
            

            float _RepulsionScale;
            float _RepulsionInfluence;
            
            float _AlignmentScale;
            float _AlignmentInfluence;

            float _GroupingScale;
            float _GroupingInfluence;

            float _BoundaryScale;
            float _BoundaryInfluence;
            float _BoundaryDistance;
            float3 _BoundaryMultipliers;


            float _ClosenessScale;
            float _ExclusionScale;


            float _RepulsionDistance;


            float _SpeedLimit;
            float _Damping;

            sampler2D _Noise;
            float _SpeedNoise;

            #define TEXTURE_WIDTH 0032.0
            #define TEXTURE_HEIGHT 0032.0
            #define BOID_COUNT 1024

            #define PI 3.14159265
            #define ROOT_HALF .707106781

            // Two Color conversion methods courtesy of 1001 from the VRC Shader community
            float3 hsv2rgb(float x, float y, float z) {
                return z + z * y * (clamp(abs(fmod(x * 6. + float3(0, 4, 2), 6.) - 3.) - 1., 0., 1.) - 1.);
            }
            float3 rgb2hsv(float3 c)
            {
                float4 K = float4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
                float4 p = lerp(float4(c.bg, K.wz), float4(c.gb, K.xy), step(c.b, c.g));
                float4 q = lerp(float4(p.xyw, c.r), float4(c.r, p.yzx), step(p.x, c.r));

                float d = q.x - min(q.w, q.y);
                float e = 1.0e-10;
                return float3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
            }

            // Two memory manipulation methods courtesy of Toocanzs, of the VRC Shader community as well
            float4 float32ToRGBA8(float x)
            {
                uint4 f = asuint(x);
                uint4 bytes = (f / uint4(0x1000000, 0x10000, 0x100, 1)) & uint4(0xff, 0xff, 0xff, 0xff);
                float4 output = float4(bytes);
                return output / 255;
            }

            float RGBA8ToFloat32(float4 f)
            {
                f *= 255;
                uint4 bytes = (uint4(f) * uint4(0x1000000, 0x10000, 0x100, 1));
                uint full = bytes.x + bytes.y + bytes.z + bytes.w;
                return asfloat(full);
            }
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

            float pMod1(inout float p, float size)
            {
                float halfsize = size * 0.5;
                float c = floor((p + halfsize) / size);
                p = fmod(abs(p) + halfsize, size) - halfsize;
                return c;

            }

            float pModSingle1(inout float p, float size)
            {
                float halfsize = size * .5;
                float c = floor((p + halfsize) / size);
                if (p >= 0)
                    p = fmod(p + halfsize, size) - halfsize;
                return c;
            }

            float pModPolar(inout float2 p, float repetitions) {
                float angle = 2 * PI / repetitions;
                float r = length(p);
                float a = (atan2(p.x, p.y) + PI) + angle * 0.5;
                float c = floor(a / angle);
                a = fmod(a, angle) - angle * 0.5;
                p = float2(cos(a), sin(a)) * r;
                if (abs(c) >= (repetitions / 2)) c = abs(c);
                return c;
            }

            float pModPolar(inout float2 p, float repetitions, float phase) {
                float angle = 2 * PI / repetitions;
                float r = length(p);
                float a = (atan2(p.x, p.y) + PI + (2 * PI * phase / 360)) + angle * 0.5;
                float c = floor(a / angle);
                a = fmod(a, angle) - angle * 0.5;
                p = float2(cos(a), sin(a)) * r;
                if (abs(c) >= (repetitions / 2)) c = abs(c);
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

            //float DE(float3 p)
            //{

            //    float3 center = float3(0, 0, 0);

            //    //pModMirror(p.y);

            //    float3 r = p;
            //    float3 s = p;
            //    float3 v = p;
            //    float3 w = p;
            //    float3 a = p;

            //    float d;

            //    pModPolar(r.xz, 8);
            //    pModPolar(s.xz, 16, 11.25);

            //    float dPillar = fPillar(r, float3(16, 10, 0), .5, 10);
            //    float dBox = fBox(s, float3(.5, .5, 10), float3(15.7, 6, 0));

            //    d = min(dBox, dPillar);

            //    float3 t = p;
            //    float3 u = p;
            //    pModPolar(t.xz, 8, 22.5);
            //    pModPolar(u.xz, 8, 22.5);
            //    pR45(t.yz, float2(4, 0));
            //    float size = 1;
            //    float radius = 1.2;
            //    float3 archCenter = float3(30, 4, 0);
            //    float dBoxA = fBox(t, float3(1, 1, 1) * size, archCenter);
            //    float dBoxA2 = fBox(u, float3(1, 2, 1.64) * size, archCenter - float3(0, 2, 0));
            //    float dSphereA = fSphere(t, radius, archCenter);
            //    float dArch = min(fOpUnionChamfer(dBoxA, dSphereA, .6), dBoxA2);

            //    pModPolar(v.xz, 16, 22.5);
            //    float dBox2 = fBox(v, float3(.8, 8, 15), float3(30, 4, 0));
            //    //dArch = max(dBox2, -dArch);
            //    d = min(d, dBox2);

            //    d = min(d, 8.5 - p.y);
            //    d = min(d, p.y + 0.75);

            //    return d;

            //}
            float DE(float3 p)
            {
                float3 z3 = float3(0, 0, 0);

                float sphere = -fSphere(p, 37.5, float3(0, 0, 0));
                //float cyl0 = fCapsule(p.zxy, 8.14, 4, float3(0, -4.2, -25.9));
                float cyl0 = fCapsule(p.zxy, 13.9, 5.77, float3(0.45, -6.58, -21.6));
                float box = fBox(p.zxy, float3(5.77, 9.04, 2.6), float3(0.45, -6.58 + 13.9 * 0.5, -21.6 + -2.92));
                float d = sphere;
                d = min(d, cyl0);
                d = smin(d, box, 5);
                //float box0 = fBox(p, float3(2.28, 2.597855, 1.827455)*1.2, float3(-0.46, -2.31, 3.01));
                //float sphere0 = fSphere(p, 1.83, float3(-4.79, -3.9 - 0.32, 3.08));
                //float cyl0 = fCapsule(p, 1.89, 0.51, float3(3.43, -2.58, 0.31));

                return d;
            }





            // First off, this shader is meant for accessing information stored in a very specific way.
            //
            // UV map diagram:
            //
            //  w   ...     ...
            //  z   ...     ...
            //  y   ...     ...
            //  x   ...     ...
            //      pos     vel
            //
            // Within the uv map for the boids, each boid is mapped to exacltly one pixel. Those pixels are all concentrated
            // by default within the (pos, x) coordinate on the grid above. Each coordinate is to act as its own
            // image, allowing one shader to output to a total of 8 simulated textures, each containing the information
            // of GPU particles corresponding to their assigned location.

            // We're going to hard code a couple getter methods for storing and rendering what part of the
            // Particle information we need. Here's the idea:
            //
            // The particles are uv-unwrapped, one boid per pixel. The uv's must be shrunken down to the bottom
            // eight grid first through, and this shrunken uv coordinate will be refered to as the base uv.
            //
            // In this method, we can first calculate what portion of the info the pixel we are in is calculating, use the
            // offsets to get the relevant information, and return the info our pixel needs. Lets get started on those methods
            //
            // First we want to get a float4 containing a 1 in the index we desire to return, this way a good dot product
            // can filter out the x, y, z, or w information as we want regardless once everything has been calculated.
            // We'll call this float4 a discriminator.


            // The incoming uvs are scale from 0-1 across the entire quad. However, we want that 0-1 range to be a set base value,
            // that covers only the bottom most corner of our grid. This way, we simulate having 8 images with linked points 
            // accessible by applying an offset value to the base uv. In our case, this is pretty simple. 
            float2 generateBaseUv(float2 uv)
            {
                // Multiplying by 4 vertically means we hit one at the first vertical quarter mark.
                // Similarly, multpilying by 2 horizontally means we hit 1/2 at the second vertical quarter mark
                // Really, this method is for readability more than anything.
                float2 scaled = uv * float2(2.0,4.0);
                float2 fracc = float2(frac(scaled.x), frac(scaled.y));
                float2 rescaled = fracc * float2(0.5, 0.25);
                
                //return (float2( int2(uv.x*32.0, uv.y*32.0) ) / 32) * float2(0.5, .25);
                //return uv * float2(0.5, 0.25);
                return rescaled;
            }

            int SimulatorUVToIndex(float2 uv)
            {
                float2 norm = frac(uv * float2(2.0, 4.0)) * float2(TEXTURE_WIDTH, TEXTURE_HEIGHT);
                return (int)(norm.x + norm.y * TEXTURE_WIDTH + 0.2);
            }


            // Generate uv coordinates for a particle of a given index.
            float2 ParticleIndexToUV(int particleIndex)
            {
                int x = particleIndex % TEXTURE_WIDTH;
                int y = particleIndex / TEXTURE_WIDTH;
                float2 uv = float2(float(x) + 0.5, float(y) + 0.5) / float2(TEXTURE_WIDTH, TEXTURE_HEIGHT);
                return uv;
            }


            //float3 safe_normalize(float3 vec)
            //{
            //    float len = length(vec);
            //    vec = normalize(vec);
            //    if (len < 0.0001)
            //    {
            //        vec = float3(0, 0, 0);
            //    }
            //    return vec;
            //}

            // Get a struct of particle data given some uv coordinates
            ParticleData SampleParticleInfo(float2 continuous_uv)
            {
                ParticleData partData;

                // Generate base uv's for the bottom corner of the uv-grid. The 'Base UV' is used as if there was only one square
                // of pixels, being the entirety of the render plane rather than divided up into our 8 groups.


                //float2 uv = ParticleIndexToUV(index);
                float2 baseUv = generateBaseUv(continuous_uv);

                float2 divisions = float2(TEXTURE_WIDTH * 2, TEXTURE_HEIGHT * 4);
                float2 uv = (floor(baseUv * divisions) + 0.5) / divisions;

                baseUv = uv;

                //Alright, now we can get to fetching the information for each gpu particle.
                //float4 vertical_offsets = float4(3.0, 2.0, 1.0, 0.0) * 0.25;
                float4 vertical_offsets = float4(0.0, 1.0, 2.0, 3.0) * 0.25;


                // First, we will generate all the uv coordinates for the information
                // Generate UV offset for sampling position and velocity
                float2 xPosUv = baseUv + float2(vertical_offsets.x, 0.0).yx;
                float2 yPosUv = baseUv + float2(vertical_offsets.y, 0.0).yx;
                float2 zPosUv = baseUv + float2(vertical_offsets.z, 0.0).yx;
                float2 wPosUv = baseUv + float2(vertical_offsets.w, 0.0).yx;

                float2 xVelUv = baseUv + float2(vertical_offsets.x, 0.5).yx;
                float2 yVelUv = baseUv + float2(vertical_offsets.y, 0.5).yx;
                float2 zVelUv = baseUv + float2(vertical_offsets.z, 0.5).yx;
                float2 wVelUv = baseUv + float2(vertical_offsets.w, 0.5).yx;

                // Edit: Swizzled because I'm an idiot and am too lazy to swap them


                // Now, we will sample for all the relavent information
                // Sample all the uv offsets for the position and velocity data (stored as color)
                float4 xPosCol = tex2D(_MainTex, xPosUv);
                float4 yPosCol = tex2D(_MainTex, yPosUv);
                float4 zPosCol = tex2D(_MainTex, zPosUv);
                float4 wPosCol = 0;

                float4 xVelCol = tex2D(_MainTex, xVelUv);
                float4 yVelCol = tex2D(_MainTex, yVelUv);
                float4 zVelCol = tex2D(_MainTex, zVelUv);
                float4 wVelCol = 0;


                // Of course, thats useless if its still stored in color form, so we'll convert to floats
                // Convert all the sampled information to float form
                float xPos = RGBA8ToFloat32(xPosCol);
                float yPos = RGBA8ToFloat32(yPosCol);
                float zPos = RGBA8ToFloat32(zPosCol);
                float wPos = 0;

                float xVel = RGBA8ToFloat32(xVelCol);
                float yVel = RGBA8ToFloat32(yVelCol);
                float zVel = RGBA8ToFloat32(zVelCol);
                float wVel = 0;


                // Now lets construct the position and velocity vectors
                partData.pos = float4(xPos, yPos, zPos, wPos);
                partData.vel = float4(xVel, yVel, zVel, wVel) * _VelocityScale;

                return partData;

            }
            
            // Get a struct of particle data given the particles index. This will be used for iterating through all the
            // boids being simulated
            ParticleData SampleSerializedParticleInfo(int index)
            {
                ParticleData partData;
                
                // Generate base uv's for the bottom corner of the uv-grid. The 'Base UV' is used as if there was only one square
                // of pixels, being the entirety of the render plane rather than divided up into our 8 groups.


                float2 uv = ParticleIndexToUV(index);
                return SampleParticleInfo(uv);

            }

            // Calculates the inverse square repulsional force between two points. Simply flip the sign to
            // make it an attracting force.
            float3 CalculateInverseSquareRepulsion(float3 point1, float3 point2)
            {
                float3 gvec = normalize(point2-point1);
                float gvec_magnitude = length(point2 - point1);

                float gfor = 1.0 / ( dot(gvec, gvec) );
                // Handle the edge case where our vector is zero, and normalization of it would result in infinities or NaN's.
                if (gvec_magnitude < 0.001)
                {
                    gfor = 0.0;
                    gvec = float3(0, 0, 0);
                }

                return gfor * gvec;
            }


            float4 safe_normalize(float4 vec)
            {
                float dp3 = max(0.01, dot(vec, vec));
                return vec * rsqrt(dp3);
            }
            float3 safe_normalize(float3 vec)
            {
                float dp3 = max(0.01, dot(vec, vec));
                return vec * rsqrt(dp3);
            }

            float SDF0(float3 pos)
            {


                return DE(pos);

                //float radius = 20;
                //float3 center = float3(0, 0, 0);
                //return -( ( length(pos - center) ) - radius );
            }

            float3 VectorField0(float3 pos)
            {
                float max_distance = _BoundaryDistance;


                float delta = 0.05;
                float2 offset = float2(1, 0) * delta;

                float3 pos_dxp = pos + offset.xyy;
                float3 pos_dxn = pos - offset.xyy;

                float3 pos_dyp = pos + offset.yxy;
                float3 pos_dyn = pos - offset.yxy;

                float3 pos_dzp = pos + offset.yyx;
                float3 pos_dzn = pos - offset.yyx;

                float dx = SDF0(pos_dxp) - SDF0(pos_dxn);
                float dy = SDF0(pos_dyp) - SDF0(pos_dyn);
                float dz = SDF0(pos_dzp) - SDF0(pos_dzn);

                float dist = SDF0(pos);
                //float invsqr = 1.0 / (dist - max_distance)
                if (dist < 0)
                {
                    dist *= 10;
                }

                float3 gradient = float3(dx, dy, dz);

                float3 vforce = gradient * max((max_distance - min(dist, max_distance)), 0);
                return vforce;
            }

            // Calculates Interboid forces, including forces of repulsion between nearby boids, the force
            // causing them to align with one another, and the force steering them towards the center of
            // the nearest handful of boids
            float3 CalculateInterboidForces(float3 pos, float3 vel, float3 noise, int id)
            {

                // initialize an empty spot for our net force
                float3 netForce = float3(0.00001, 0.0, 0.0);

                float3 up = float3(0, 1, 0);
                float3 right = -normalize(cross(vel, up));
                up = normalize(cross(vel, right));

                float delta = 0.15;
                float3 offset_right = right * delta;
                float3 offset_up = up * delta;

                float3 pos_dxp = pos + vel * 0.5 + offset_right * length(vel);
                float3 pos_dxn = pos + vel * 0.5 - offset_right * length(vel);

                float3 pos_dyp = pos + vel * 0.5 + offset_up;
                float3 pos_dyn = pos + vel * 0.5 - offset_up;

                float3 pos_fw = pos + vel * 0.5;

                float min_dxp = 1000;
                float min_dxn = 1000;

                float min_dyp = 1000;
                float min_dyn = 1000;

                float3 center_mass = float3(0, 0, 0);
                float center_mass_boid_count = 0.0;

                float3 alignment_vector = float3(0, 0, 0);

                int boid_count = TEXTURE_WIDTH * TEXTURE_HEIGHT;

                int sample_set = id & 7;
                float4 zer = float4(0, 0, 0, 0);
                float4x4 boid_matrix = float4x4(zer, zer, zer, zer);
                float4 min_dists = float4(1000, 1000, 1000, 1000);


                // Go through all 1024 boids. We will denote the boid we are calculating the force for as our boid, and the boids we are 
                // iterating through as particles for ease of reference.
                for (int i = 0; i < boid_count; i = i + 4)
                {
                    ParticleData particle = SampleSerializedParticleInfo(i + sample_set);

                    bool canBoidNotBeSeen = (distance(particle.pos, pos) > 5.0) || (distance(particle.pos, pos) < 0.000000048828125) || (dot(particle.pos - pos, vel) < -0.21);

                    if (length(particle.pos.xyz - pos_fw) < min_dists.x)
                    {
                        float4 ones = float4(1, 1, 1, 1);
                        float4 zeros = float4(0, 0, 0, 0);

                        boid_matrix[3] = boid_matrix[2];
                        boid_matrix[2] = boid_matrix[1];
                        boid_matrix[1] = boid_matrix[0];
                        boid_matrix[0] = particle.pos;

                        min_dists = min_dists.wxyz;
                        min_dists.x = distance(particle.pos, pos);
                    }

                    if (!canBoidNotBeSeen)
                    {

                        min_dxp = min(min_dxp, distance(particle.pos, pos_dxp));
                        min_dxn = min(min_dxn, distance(particle.pos, pos_dxn));
                        min_dyp = min(min_dyp, distance(particle.pos, pos_dyp));
                        min_dyn = min(min_dyn, distance(particle.pos, pos_dyn));

                        center_mass += particle.pos - pos_fw;
                        center_mass_boid_count += 1.0;

                        alignment_vector += particle.vel;
                    }
                }

                float3 p0 = boid_matrix[0].xyz;
                float3 p1 = boid_matrix[1].xyz;
                float3 p2 = boid_matrix[2].xyz;
                float3 p3 = boid_matrix[3].xyz;

                float r = _RepulsionDistance;
                float d0 = min(min_dists.x - r, 0.001);
                float d1 = min(min_dists.y - r, 0.001);
                float d2 = min(min_dists.z - r, 0.001);
                float d3 = min(min_dists.w - r, 0.001);

                float d0_2 = min_dists.x * min_dists.x * min_dists.x;
                float d1_2 = min_dists.y * min_dists.y * min_dists.y;
                float d2_2 = min_dists.z * min_dists.z * min_dists.z;
                float d3_2 = min_dists.w * min_dists.w * min_dists.w;

                float3 v0 = (pos - p0) * (1.0 / d0_2);
                float3 v1 = (pos - p1) * (1.0 / d1_2);
                float3 v2 = (pos - p2) * (1.0 / d2_2);
                float3 v3 = (pos - p3) * (1.0 / d3_2);

                float3 displacement = v0 + v1 + v2 + v3;


                float uniqueOffset = id / float(boid_count);
                float uniqueTheta = (id * 5 * 1.118);
                float2 uniqueDirection = float2(cos(uniqueTheta), sin(uniqueTheta)) * .00075;

                center_mass *= (1.0 / max(1, center_mass_boid_count));
                alignment_vector *= (1.0 / max(1, center_mass_boid_count));

                //float3 grouping_projected = vel - center_mass * ( dot(center_mass, vel) );
                float3 grouping_projected = center_mass + uniqueDirection.x * right + uniqueDirection.y * up;

                float grouping_length = length(grouping_projected);
                float3 grouping_force = normalize(grouping_projected) * _GroupingScale * (1.0 / min(grouping_length * grouping_length, 0.1));
                if (grouping_length < 0.001)
                {
                    grouping_force = 0.0;
                }

                float dx = min_dxp - min_dxn;
                float dy = min_dyp - min_dyn;

                float min_dist = min(min(min_dxp, min_dxn), min(min_dyp, min_dyn));


                float repulsion_radius = _RepulsionDistance;
                float2 gradient = float2(min_dxp - min_dxn, min_dyp - min_dyn);

                float2 vforce = gradient * max((repulsion_radius - min(min_dist, repulsion_radius)), 0);
                //float2 vforce = gradient * min( max(   1.0 / ( min_dist*min_dist ), 0), 100);
                float3 repulsion_force = (vforce.x * right + vforce.y * up) * _RepulsionScale;//; * max(1, center_mass_boid_count;
                //repulsion_force += cross(repulsion_force, vel);





                float3 alignment_force = alignment_vector * _AlignmentScale + (uniqueDirection.x * right + uniqueDirection.y * up) * 500;


                float3 bforce = VectorField0(pos.xyz) * _BoundaryScale * _BoundaryMultipliers;
                //bforce += cross(bforce, vel);// *saturate(1 - dot(vel, bforce));

                float acceleration = 0.01;
                float3 new_vel = vel;
                new_vel = lerp(new_vel, vel + grouping_force, _GroupingInfluence);
                new_vel = lerp(new_vel, vel + alignment_force, _AlignmentInfluence);
                //new_vel = lerp(new_vel, vel + repulsion_force, _RepulsionInfluence);
                //new_vel = lerp(new_vel, vel + bforce, _BoundaryInfluence);

                new_vel = lerp(new_vel, vel + grouping_force * _GroupingInfluence + alignment_force * _AlignmentInfluence, saturate(_GroupingInfluence + _AlignmentInfluence));
                new_vel = lerp(new_vel, vel + repulsion_force, _RepulsionInfluence);
                new_vel = lerp(new_vel, vel + bforce, _BoundaryInfluence);


                return new_vel;

                //netForce = grouping_force;
                netForce = grouping_force + alignment_force + repulsion_force;

                // return our sum of all the differential forces as a net force
                return netForce;
            }







            // Calculate the repulsionary force that will steer the boids to be within some specified cube
            float3 CalculateBoundaryRepulsion(float3 p, float cubeSize)
            {
                //float3 fvec = float3(0, 0, 0);
                //float3 offset = p;
                //if (abs(offset.x) > cubeSize)
                //{
                //    fvec.z = -offset.x*abs(offset.x);
                //}
                //if (abs(offset.y) > cubeSize)
                //{
                //    fvec.y = -offset.y*abs(offset.y);
                //}
                //if (abs(offset.z) > cubeSize)
                //{
                //    fvec.z = -offset.z*abs(offset.z);
                //}
                //return fvec;


                float3 fvec = float3(0, 0, 0);
                float3 offset = p;
                if (abs(offset.x) > cubeSize*0.5 || abs(offset.y) > cubeSize * 0.5 || abs(offset.z) > cubeSize * 0.5)
                {
                    fvec = -offset*length(offset);
                }
                return fvec;

            }


            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = TRANSFORM_TEX(v.uv, _MainTex);
                return o;
            }




            fixed4 frag(v2f i) : SV_Target
            {
                // Generate some discriminator based on the uv. This creates a float4 that strips
                // out all but some desired index of information from a value. That info is determined
                // by the range of the v coordinate passed in.

                // too lazy to specify our v2f as the object owning the uv, so lets declare a new float2.
                // This'll be compiled out anyway.
                
                float2 divisions = float2(TEXTURE_WIDTH*2, TEXTURE_HEIGHT*4);
                float2 uv = (floor(i.uv * divisions) + float2(0.5, 0.5) ) / divisions;

                float v0 = (uv.y > 0.00) && (uv.y < 0.25);
                float v1 = (uv.y > 0.25) && (uv.y < 0.50);
                float v2 = (uv.y > 0.50) && (uv.y < 0.75);
                float v3 = (uv.y > 0.75) && (uv.y < 1.00);

                float4 discriminator = float4(v0, v1, v2, 0.0);
                


                // Sample our current boid's info based on the incoming uv coordinate.
                ParticleData particle = SampleParticleInfo(i.uv);

                // Now lets construct the position and velocity vectors
                float4 particle_pos = particle.pos;
                float4 particle_vel = particle.vel;

                // Now lets update the particles' positions based on velocity. First, scale this down to match the target frame rate, then
                // scale it back up according to user specification


                ////////////////////////////////////////////////////////
                ////              Particle Simulation               ////
                ////////////////////////////////////////////////////////

                // sample some noise coloe
                float3 noise_color = tex2D(_Noise, i.uv).xyz;

                // Lets start with a base time scale, specified by the user, divided to match the target frame rate
                float time_scale = _TimeScale * (1.0 / 90.0);

                //float2 upscaled_uv = frac(uv * float2(2.0, 4.0));


                int boid_id_from_pixel = SimulatorUVToIndex(uv);

                // Calculate all the forces acting on our boid.
                //float3 gfor = CalculateInverseSquareRepulsion(worldTarget.xyz, particle_pos.xyz);
                float3 new_vel = CalculateInterboidForces(particle_pos.xyz, particle_vel.xyz, noise_color.xyz, boid_id_from_pixel);
                new_vel = lerp(particle_vel.xyz, new_vel, _TimeScale);

                //rfor = float3(0, 0, 0);
                // Generate the acceleration and velocity vectors. Gfor and bfor are adjustable by parameters. rfor was the repulsive force, but is now any
                // interboid force, and because any interboid force requires iterating through all boids, they were placed in the same method for optimzation
                //purposes. they are adjusted via parameter inside the method.

                // TODO: Make parameter adjustments consistent, either by moving the parameters to the interior of the gfor and bfor calculation methods,
                // or return the interboid forces as a struct. Basically, anything to make this more consistent.
                //float3 a_vec = rfor;
                float3 v_vec = particle_vel;
                particle_vel = float4(new_vel, 0);

                // For consistency of simulation results, lets limit it to double for now.
                float frameComp = clamp(unity_DeltaTime * 90, 0.25, 2.5);

                // update our position and velocities.
                particle_pos = lerp(particle_pos, particle_pos + float4(v_vec, 0.0) * _VelocityScale * frameComp * (1.00+0.15*noise_color.x) * .01, _TimeScale);
                //particle_vel += float4(a_vec, 0.0) * _TimeScale * frameComp * (1.00+0.15*noise_color.x);
                

                // Clamp our velocity to be within a certain length. This is mainly because the velocity kept reaching zero
                // (resulting in divisions by zero in normalizing vectors of length 0) or exploding off towards infinity.
                // Really, we want our boids going approximately the same velocity as well.
                float4 pvel_norm = normalize(particle_vel);
                pvel_norm.y = clamp(pvel_norm.y, -0.95, 0.95);
                pvel_norm.xz += float2(_CosTime.w, _SinTime.w) * saturate(1 - length(pvel_norm.xz) * 4)*0.1;
                pvel_norm = normalize(pvel_norm);

                particle_vel = (clamp(length(particle_vel), 0.25, 1.00))  * pvel_norm + float4(noise_color - float3(0.5,0.5,0.5), 0)*_AttractionScale;




                // Here's where the magic discriminator comes in. we don't actually need to convert everything back into a color to get across
                // the information we need. Lets first strip out the relevant velocities and positions
                float return_pos = dot(discriminator, particle_pos);
                float return_vel = dot(discriminator, particle_vel);
                
                // Generate a fixed value for interpolation between position and value. This will be done using a step function and the original uv.x
                // coordinate. u_(1/2) for those unfamiliar is a function that 'steps' up at 0.5. It is zero before, and 1.0 after. the second term
                // i.uv.x is used as the functions input value.
                fixed isVel = step(0.5, i.uv.x);

                // Lerp between the position and velocity with the isVel psuedo-boolean as the factor.
                float return_val = 0.0;

                if (isVel >= 0.5)
                {
                    return_val = return_vel;
                }
                else
                {
                    return_val = return_pos;
                }

                // convert the return value to a color
                fixed4 col = float32ToRGBA8(return_val);

                // return that color.
                return col;
            }
            ENDCG
        }
    }
}