Shader "SilVR/Boids/Mesh Boids Fish"
{
    Properties
  {

        _MainTex("Render Texture", 2D) = "white" {}
        _FishTex("Fish Texture", 2D) = "white" {}
        _Noise("Texture", 2D) = "white" {}

        _UpNoiseScale("Vector up noise", float) = .2
        _UpNoiseSpeed("Vector up noise speed", float) = .22

        _WorldScale("World Scale", float) = 10.0
        _ModelScale("Model Scale", float) = 1.63
        _SizeNoise("Size noise", float) = 0.43

        _Debug("Debug Range", Range(0,1)) = 0

        _WaveScale("Wave scale", float) = 0.25
        _WaveFrequency("Wave frequency", float) = 1
        _WaveLength("Wave length", float) = 4.14
        _WaveNoise("Wave noise", float) = 60.94

    }
    SubShader
    {




        Pass
        {
            Tags { "RenderType" = "Opaque" }
            LOD 100
            //Cull Off

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
                float2 uv1 : TEXCOORD1;
                float4 col : COLOR;
            };

            struct v2f
            {
                float2 uv : TEXCOORD0;
                float2 uv1 : TEXCOORD1;
                float4 vertex : SV_POSITION;
                float4 wvel : TEXCOORD2;
                float4 boid_offset : TEXCOORD3;
                float4 col : COLOR;
            };

            struct ParticleData
            {
                float4 pos;
                float4 vel;
            };


            sampler2D _MainTex;
            float4 _MainTex_ST;

            sampler2D _FishTex;
            sampler2D _Noise;

            float _UpNoiseScale;
            float _UpNoiseSpeed;

            float _WorldScale;
            float _ModelScale;
            float _SizeNoise;

            float _WaveScale;
            float _WaveLength;
            float _WaveFrequency;
            float _WaveNoise;

            float _Debug;


            //float2 generateBaseUv(float2 uv)
            //{
            //    // Multiplying by 4 vertically means we hit one at the first vertical quarter mark.
            //    // Similarly, multpilying by 2 horizontally means we hit 1/2 at the second vertical quarter mark
            //    // Really, this method is for readability more than anything.
            //    float2 scaled = uv * float2(2.0, 4.0);
            //    float2 fracc = float2(frac(scaled.x), frac(scaled.y));
            //    float2 rescaled = fracc * float2(0.5, 0.25);

            //    return uv * float2(0.5, .25);
            //}

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

            ParticleData SampleParticleInfo(float2 uv)
            {
                ParticleData partData;

                // Generate base uv's for the bottom corner of the uv-grid. The 'Base UV' is used as if there was only one square
                // of pixels, being the entirety of the render plane rather than divided up into our 8 groups.


                //float2 uv = ParticleIndexToUV(index);
                float2 baseUv = uv*float2(0.5, 0.25);

                //Alright, now we can get to fetching the information for each gpu particle.
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
                float4 xPosCol = tex2Dlod(_MainTex, float4(xPosUv, 0, 0));
                float4 yPosCol = tex2Dlod(_MainTex, float4(yPosUv, 0, 0));
                float4 zPosCol = tex2Dlod(_MainTex, float4(zPosUv, 0, 0));
                float4 wPosCol = tex2Dlod(_MainTex, float4(wPosUv, 0, 0));

                float4 xVelCol = tex2Dlod(_MainTex, float4(xVelUv, 0, 0));
                float4 yVelCol = tex2Dlod(_MainTex, float4(yVelUv, 0, 0));
                float4 zVelCol = tex2Dlod(_MainTex, float4(zVelUv, 0, 0));
                float4 wVelCol = tex2Dlod(_MainTex, float4(wVelUv, 0, 0));


                // Of course, thats useless if its still stored in color form, so we'll convert to floats
                // Convert all the sampled information to float form
                float xPos = RGBA8ToFloat32(xPosCol);
                float yPos = RGBA8ToFloat32(yPosCol);
                float zPos = RGBA8ToFloat32(zPosCol);
                float wPos = RGBA8ToFloat32(wPosCol);

                float xVel = RGBA8ToFloat32(xVelCol);
                float yVel = RGBA8ToFloat32(yVelCol);
                float zVel = RGBA8ToFloat32(zVelCol);
                float wVel = RGBA8ToFloat32(wVelCol);


                // Now lets construct the position and velocity vectors
                partData.pos = float4(xPos, yPos, zPos, wPos);
                partData.vel = float4(xVel, yVel, zVel, wVel);

                return partData;

            }

            v2f vert (appdata v)
            {
                v2f o;
                o.uv = TRANSFORM_TEX(v.uv , _MainTex);

                float3 color_noise = tex2Dlod(_Noise, float4(v.uv1, 0, 0));

                ParticleData particle = SampleParticleInfo(v.uv1);

                float4 worldPos = particle.pos;
                float4 worldVel = particle.vel;

                float3 worldZero = mul(unity_WorldToObject, float4(0, 0, 0, 1)).xyz;

                float4 localPos = worldPos;
                float4 localVel = worldVel;

                float theta = _Time.w*_UpNoiseSpeed + color_noise.z;
                float3 up_noise = float3(sin(theta), 0, cos(theta));


                float3 fw = normalize(worldVel.xyz);
                float3 up = normalize(mul((float3x3)unity_WorldToObject, float3(0, 1, 0) + up_noise*_UpNoiseScale));
                float3 rt = normalize(cross(up, fw));
                up = -normalize(cross(rt, fw));

                float3 localVertexPos = -rt * v.vertex.x + up * v.vertex.z + fw * v.vertex.y;

                float wavyNoise = color_noise.x * _WaveNoise;
                float wavy = sin(_Time.w*_WaveFrequency + localVertexPos.z*_WaveLength + wavyNoise ) * _WaveScale;

                float sizeNoise = color_noise.y * _SizeNoise;
                float4 vertex = float4(worldPos.xyz*_WorldScale + localVertexPos *_ModelScale * (1.0 + sizeNoise) + rt*wavy, 1);


                o.vertex = UnityObjectToClipPos(vertex);

                float2 uv = o.uv;
                float2 uv_data = v.uv1;
                
                o.uv1 = v.uv1;
                o.wvel = localVel*100.0;
                o.boid_offset = float4(0,0,0,0);
                o.col = float4(normalize(worldVel.xyz), 1.0);

                return o;
            }
            float _VelocityScale;

            
            fixed4 frag(v2f i) : SV_Target
            {
                float2 uv_modular = float2((int2(i.uv * 32.0))) / 32.0 * float2(0.5, .25);
                fixed4 color = tex2D(_Noise, i.uv1);

                float4 col = float4(color);
                return tex2D(_FishTex, i.uv) * lerp(float4(1,1,1,1), color, .5);
                //return tex2D(_FishTex, i.uv) * color;
                
                
                //return float4(debug_color.xyz, 0.5);
                //return float4(lerp(float3(1,1,1)*0.2 + col.xyz*0.8, col.xyz*color, step(dot(i.col.xyz, float3(1,1,1)), 1.8)), 0.5);
            }
            ENDCG
        }


        Pass
        {

            Tags { "Lightmode" = "ShadowCaster" }
            LOD 100
            //Cull Off
            CGPROGRAM

            #pragma vertex vert
            #pragma fragment frag
            #pragma multi_compile_shadowcaster

            // make fog work

            #include "UnityCG.cginc"

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
                float2 uv1 : TEXCOORD1;
                float4 col : COLOR;
            };

            struct v2f
            {
                V2F_SHADOW_CASTER;
            };

            struct ParticleData
            {
                float4 pos;
                float4 vel;
            };

            sampler2D _MainTex;
            float4 _MainTex_ST;

            sampler2D _FishTex;
            sampler2D _Noise;

            float _UpNoiseScale;
            float _UpNoiseSpeed;

            float _WorldScale;
            float _ModelScale;
            float _SizeNoise;

            float _WaveScale;
            float _WaveLength;
            float _WaveFrequency;
            float _WaveNoise;

            float _Debug;

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
            ParticleData SampleParticleInfo(float2 uv)
            {
                ParticleData partData;

                // Generate base uv's for the bottom corner of the uv-grid. The 'Base UV' is used as if there was only one square
                // of pixels, being the entirety of the render plane rather than divided up into our 8 groups.


                //float2 uv = ParticleIndexToUV(index);
                float2 baseUv = uv * float2(0.5, 0.25);

                //Alright, now we can get to fetching the information for each gpu particle.
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
                float4 xPosCol = tex2Dlod(_MainTex, float4(xPosUv, 0, 0));
                float4 yPosCol = tex2Dlod(_MainTex, float4(yPosUv, 0, 0));
                float4 zPosCol = tex2Dlod(_MainTex, float4(zPosUv, 0, 0));
                float4 wPosCol = tex2Dlod(_MainTex, float4(wPosUv, 0, 0));

                float4 xVelCol = tex2Dlod(_MainTex, float4(xVelUv, 0, 0));
                float4 yVelCol = tex2Dlod(_MainTex, float4(yVelUv, 0, 0));
                float4 zVelCol = tex2Dlod(_MainTex, float4(zVelUv, 0, 0));
                float4 wVelCol = tex2Dlod(_MainTex, float4(wVelUv, 0, 0));


                // Of course, thats useless if its still stored in color form, so we'll convert to floats
                // Convert all the sampled information to float form
                float xPos = RGBA8ToFloat32(xPosCol);
                float yPos = RGBA8ToFloat32(yPosCol);
                float zPos = RGBA8ToFloat32(zPosCol);
                float wPos = RGBA8ToFloat32(wPosCol);

                float xVel = RGBA8ToFloat32(xVelCol);
                float yVel = RGBA8ToFloat32(yVelCol);
                float zVel = RGBA8ToFloat32(zVelCol);
                float wVel = RGBA8ToFloat32(wVelCol);


                // Now lets construct the position and velocity vectors
                partData.pos = float4(xPos, yPos, zPos, wPos);
                partData.vel = float4(xVel, yVel, zVel, wVel);

                return partData;

            }

            v2f vert(appdata v)
            {
                v2f o; 

                float3 color_noise = tex2Dlod(_Noise, float4(v.uv1, 0, 0));

                ParticleData particle = SampleParticleInfo(v.uv1);

                float4 worldPos = particle.pos;
                float4 worldVel = particle.vel;

                float3 worldZero = mul(unity_WorldToObject, float4(0, 0, 0, 1)).xyz;

                float4 localPos = worldPos;
                float4 localVel = worldVel;

                float theta = _Time.w * _UpNoiseSpeed + color_noise.z;
                float3 up_noise = float3(sin(theta), 0, cos(theta));


                float3 fw = normalize(worldVel.xyz);
                float3 up = normalize(mul((float3x3)unity_WorldToObject, float3(0, 1, 0) + up_noise * _UpNoiseScale));
                float3 rt = normalize(cross(up, fw));
                up = -normalize(cross(rt, fw));

                float3 localVertexPos = -rt * v.vertex.x + up * v.vertex.z + fw * v.vertex.y;

                float wavyNoise = color_noise.x * _WaveNoise;
                float wavy = sin(_Time.w * _WaveFrequency + localVertexPos.z * _WaveLength + wavyNoise) * _WaveScale;

                float sizeNoise = color_noise.y * _SizeNoise;
                float4 vertex = float4(worldPos.xyz * _WorldScale + localVertexPos * _ModelScale * (1.0 + sizeNoise) + rt * wavy, 1);
                v.vertex = vertex;

                TRANSFER_SHADOW_CASTER(o);
                return o;
            }
            float _VelocityScale;


            fixed4 frag(v2f i) : SV_Target
            {
                SHADOW_CASTER_FRAGMENT(i);
            }
            ENDCG
        }

    }
}
