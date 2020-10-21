Shader "SilVR/Boids/Boid Mesh Visualizer"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
        _Noise("Texture", 2D) = "white" {}
        _Scale("Scale", float) = 1.0
        _TrailLength("Trail Length", float) = 1.0

        _Color1("Color 1", Color) = (1,1,1,1)
        _Color2("Color 2", Color) = (1,1,1,1)
        _IsTex("Is Textured", Range(0,1) ) = 1
        _IsCol("Is Color", Range(0,1) ) = 1
        _VelocityScale("Velocity Scale", float) = 1

        _ModelScale("Model Scale", float) = 1

        _Debug("Debug Range", Range(0,1)) = 0

    }
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        LOD 100
        Cull Off

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma geometry geom

            // make fog work
            #pragma multi_compile_fog

            #include "UnityCG.cginc"
            #include "ShaderUtils.cginc"
            

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
                float4 col : COLOR;
            };

            struct v2g
            {
                float2 uv : TEXCOORD0;
                //float2 uvScaled : TEXCOORD3;

                float4 vertex : SV_POSITION;
                float4 wvel : TEXCOORD1;
                float4 boid_offset : TEXCOORD2;
                float4 col : COLOR;
            };


            struct g2f
            {
                float2 uv : TEXCOORD0;
                float4 vertex : SV_POSITION;
                float wvel : TEXCOORD1;
                float tint : TEXCOORD2;
                float4 col : COLOR;

                float4 debug :  TEXCOORD4;

            };

            struct ParticleData
            {
                float4 pos;
                float4 vel;
            };

            sampler2D _MainTex;
            sampler2D _Noise;
            float4 _MainTex_ST;
            float _Scale;
            float _TrailLength;

            float _ModelScale;
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

            v2g vert (appdata v)
            {
                v2g o;
                o.uv = TRANSFORM_TEX(v.uv , _MainTex);

                float2 uvRaw = o.uv;

                ParticleData particle = SampleParticleInfo(o.uv);

                float4 worldPos = particle.pos;
                float4 worldVel = particle.vel;

                float3 worldZero = mul(unity_WorldToObject, float4(0, 0, 0, 1)).xyz;

                float4 localPos = worldPos;
                float4 localVel = worldVel;

                o.wvel = localVel*100.0;
                o.vertex = v.vertex;
                o.boid_offset = localPos * _Scale;
                o.col = v.col;



                return o;
            }
            float _VelocityScale;

            [maxvertexcount(12)]
            void geom(triangle v2g IN[3], inout TriangleStream<g2f> triStream)
            {
                g2f o;


                

                //float4 root_pos = 0.0;
                
                float4 root_pos = IN[0].vertex;

                float4 v0 = float4( float3(0, 0, 0) - float3(0.5, 0.5, 0.0), 0.0) * 0.0075;
                float4 v1 = float4( float3(1, 0, 0) - float3(0.5, 0.5, 0.0), 0.0) * 0.0075;
                float4 v2 = float4( float3(0, 1, 0) - float3(0.5, 0.5, 0.0), 0.0) * 0.0075;
                float4 v3 = float4( float3(1, 1, 0) - float3(0.5, 0.5, 0.0), 0.0) * 0.0075;
                float4 v4 = float4( float3(0, 0, 1) - float3(0.5, 0.5, 0.0), 0.0) * 0.0075;
                float4 v5 = float4( float3(1, 0, 1) - float3(0.5, 0.5, 0.0), 0.0) * 0.0075;
                float4 v6 = float4( float3(0, 1, 1) - float3(0.5, 0.5, 0.0), 0.0) * 0.0075;
                float4 v7 = float4( float3(1, 1, 1) - float3(0.5, 0.5, 0.0), 0.0) * 0.0075;

                float2 uv0Scaled = IN[0].uv * 32.0;
                float2 uv1Scaled = IN[1].uv * 32.0;
                float2 uv2Scaled = IN[2].uv * 32.0;

                float2 uv0Frac = float2(frac(uv0Scaled.x), frac(uv0Scaled.y));
                float2 uv1Frac = float2(frac(uv1Scaled.x), frac(uv1Scaled.y));
                float2 uv2Frac = float2(frac(uv2Scaled.x), frac(uv2Scaled.y));

                float3 top_disc = float3(float(uv0Frac.y > 0.5), float(uv1Frac.y > 0.5), float(uv2Frac.y > 0.5));
                float3 right_disc = float3(float(uv0Frac.x > 0.6), float(uv1Frac.x > 0.6), float(uv2Frac.x > 0.6));
                float3 left_disc = float3(float(uv0Frac.x < 0.4), float(uv1Frac.x < 0.4), float(uv2Frac.x < 0.4));

                float4 top = top_disc.x * IN[0].vertex + top_disc.y * IN[1].vertex + top_disc.z * IN[2].vertex;
                float4 right = right_disc.x * IN[0].vertex + right_disc.y * IN[1].vertex + right_disc.z * IN[2].vertex;
                float4 left = left_disc.x * IN[0].vertex + left_disc.y * IN[1].vertex + left_disc.z * IN[2].vertex;

                //float3 wx = mul(unity_ObjectToWorld, IN[0].vertex);
                //float3 wy = mul(unity_ObjectToWorld, IN[1].vertex);
                //float3 wz = mul(unity_ObjectToWorld, IN[2].vertex);
                //

                float3 wx = IN[0].vertex;
                float3 wy = IN[1].vertex;
                float3 wz = IN[2].vertex;

                float3 normal = -normalize(cross(wx - wy, wy - wz));

                float tint = (1 + dot(normal, normalize(float3(.15, -1.0, .15))))*0.75;




                float2 uv0 = IN[0].uv;
                float2 uv1 = IN[1].uv;
                float2 uv2 = IN[2].uv;

                bool uv0y_largest = uv0.y > uv1.y && uv0.y > uv2.y;
                bool uv1y_largest = uv1.y > uv2.y && uv1.y > uv0.y;
                bool uv2y_largest = uv2.y > uv0.y && uv2.y > uv1.y;


                float4 bot = (right + left) * 0.5;
                float height = length(top.xyz - bot.xyz) * _ModelScale;

                float4 up = normalize(mul(unity_WorldToObject, float4(0, 1, 0, 0)));

                float4 offset = normalize(IN[0].wvel*_VelocityScale + up*0.0001*height) * height;
                
                float3 rt = normalize(cross(up, offset));


                float3 local_up = -normalize(cross(rt, offset));
                float3 local_right = rt;
                float3 local_fw = normalize(offset.xyz);

                float triangleWidth = 0.004;

                float3 vertex0 = (IN[0].vertex.x * local_right + IN[0].vertex.y * local_fw + IN[0].vertex.z * local_up)* _ModelScale + IN[0].boid_offset;
                float3 vertex1 = (IN[1].vertex.x * local_right + IN[1].vertex.y * local_fw + IN[1].vertex.z * local_up)* _ModelScale + IN[0].boid_offset;
                float3 vertex2 = (IN[2].vertex.x * local_right + IN[2].vertex.y * local_fw + IN[2].vertex.z * local_up)* _ModelScale + IN[0].boid_offset;

                vertex0 = lerp(vertex0, float3(IN[0].uv * 100, 0), _Debug);
                vertex1 = lerp(vertex1, float3(IN[1].uv * 100, 0), _Debug);
                vertex2 = lerp(vertex2, float3(IN[2].uv * 100, 0), _Debug);

                //vertex0 = lerp(float3(IN[0].uvScaled * 100, 0), float3(IN[0].uv * 100, 0), 1-_Debug);
                //vertex1 = lerp(float3(IN[0].uvScaled * 100, 0), float3(IN[1].uv * 100, 0), 1-_Debug);
                //vertex2 = lerp(float3(IN[0].uvScaled * 100, 0), float3(IN[2].uv * 100, 0), 1-_Debug);

                

                o.debug = IN[0].boid_offset;
                o.tint = tint;
                o.vertex = UnityObjectToClipPos(vertex0);
                o.uv = TRANSFORM_TEX(IN[0].uv, _MainTex);
                o.wvel = length(IN[0].wvel);
                o.col = IN[0].col;
                triStream.Append(o);


                o.debug = IN[0].boid_offset;
                o.tint = tint;
                o.vertex = UnityObjectToClipPos(vertex1);
                o.uv = TRANSFORM_TEX(IN[1].uv, _MainTex);
                o.wvel = length(IN[0].wvel);
                o.col = IN[1].col;
                triStream.Append(o);

                o.debug = IN[0].boid_offset;
                o.tint = tint;
                o.vertex = UnityObjectToClipPos(vertex2);
                o.uv = TRANSFORM_TEX(IN[2].uv, _MainTex);
                o.wvel = length(IN[2].wvel);
                o.col = IN[2].col;
                triStream.Append(o);

                triStream.RestartStrip();
               



            }

            float4 _Color1;
            float4 _Color2;
            float _IsTex;
            float _IsCol;

            fixed4 frag(g2f i) : SV_Target
            {
                float2 uv_modular = float2((int2(i.uv * 32.0))) / 32.0 * float2(0.5, .25);

                //float4 col_pure = lerp(_Color1, _Color2, i.wvel*_VelocityScale);
                fixed4 color = tex2D(_Noise, uv_modular);

                //col_pure = lerp(float4(1, 1, 1, 1), col_pure, _IsCol);
                //col_text = lerp(float4(1, 1, 1, 1), col_text, _IsTex);

                //float4 col = col_pure * col_text;

                //float4 col = lerp(_Color1, _Color2, tex2D(_Noise, i.uv).x);

                float4 col = float4(color);

                if (dot(i.col.xyz, float3(1, 1, 1)) > 2.95)
                {
                    
                }
                float4 debug_color = i.debug;

                //return float4(debug_color.xyz, 0.5);
                return float4(lerp(float3(1,1,1)*0.2 + col.xyz*0.8, col.xyz*color, step(dot(i.col.xyz, float3(1,1,1)), 1.8)), 0.5)*i.tint;
            }
            ENDCG
        }
    }
}
