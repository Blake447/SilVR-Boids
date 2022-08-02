Shader "SilVR/Boids/Boid Visualizer"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
        _Noise("Texture", 2D) = "white" {}
        _Scale("Scale", Range(0,1)) = 1.0
        _TrailLength("Trail Length", float) = 1.0

        _Color1("Color 1", Color) = (1,1,1,1)
        _Color2("Color 2", Color) = (1,1,1,1)
        _IsTex("Is Textured", Range(0,1) ) = 1
        _IsCol("Is Color", Range(0,1) ) = 1
        _VelocityScale("Velocity Scale", float) = 1

        _ModelScale("Model Scale", float) = 1
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
            };

            struct v2g
            {
                float2 uv : TEXCOORD0;
                float4 vertex : SV_POSITION;
                float4 wvel : TEXCOORD1;

            };


            struct g2f
            {
                float2 uv : TEXCOORD0;
                float4 vertex : SV_POSITION;
                float wvel : TEXCOORD1;
                float tint : TEXCOORD2;

            };

            sampler2D _MainTex;
            sampler2D _Noise;
            float4 _MainTex_ST;
            float _Scale;
            float _TrailLength;

            float _ModelScale;

            v2g vert (appdata v)
            {
                v2g o;
                o.uv = TRANSFORM_TEX(v.uv, _MainTex);

                float2 uvRaw = o.uv;
                float2 uvScaled = float2((int2(o.uv * 32.0)))/32.0 * float2(0.5, .25);

                float4 baseUv = float4(uvScaled, 0, 0);
                float4 dataIndex = float4(0.0, 0.25, 0.0, 0.0);
                float4 posVelIndex = float4(0.5, 0.0, 0.0, 0.0);

                float4 xPosUv = baseUv + 0.0 * dataIndex + 0.0 * posVelIndex;
                float4 yPosUv = baseUv + 1.0 * dataIndex + 0.0 * posVelIndex;
                float4 zPosUv = baseUv + 2.0 * dataIndex + 0.0 * posVelIndex;
                float4 wPosUv = baseUv + 3.0 * dataIndex + 0.0 * posVelIndex;

                float4 xVelUv = baseUv + 0.0 * dataIndex + 1.0 * posVelIndex;
                float4 yVelUv = baseUv + 1.0 * dataIndex + 1.0 * posVelIndex;
                float4 zVelUv = baseUv + 2.0 * dataIndex + 1.0 * posVelIndex;
                float4 wVelUv = baseUv + 3.0 * dataIndex + 1.0 * posVelIndex;


                float xPos = RGBA8ToFloat32(tex2Dlod(_MainTex, xPosUv));
                float yPos = RGBA8ToFloat32(tex2Dlod(_MainTex, yPosUv));
                float zPos = RGBA8ToFloat32(tex2Dlod(_MainTex, zPosUv));
                float wPos = RGBA8ToFloat32(tex2Dlod(_MainTex, wPosUv));

                float xVel = RGBA8ToFloat32(tex2Dlod(_MainTex, xVelUv));
                float yVel = RGBA8ToFloat32(tex2Dlod(_MainTex, yVelUv));
                float zVel = RGBA8ToFloat32(tex2Dlod(_MainTex, zVelUv));
                float wVel = RGBA8ToFloat32(tex2Dlod(_MainTex, wVelUv));

                float4 worldPos = float4(xPos, yPos, zPos, wPos);
                float4 worldVel = float4(xVel, yVel, zVel, wVel);
                

                float3 worldZero = mul(unity_WorldToObject, float4(0, 0, 0, 1)).xyz;

                //float4 localPos = mul(unity_WorldToObject, worldPos);
                //float4 localVel = mul(unity_WorldToObject, worldVel);

                float4 localPos = worldPos;
                float4 localVel = worldVel;

                o.wvel = localVel*100.0;
                o.vertex = v.vertex + localPos*_Scale + float4(worldZero, 0.0)*0.0;

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

                //o.vertex = UnityObjectToClipPos(bot + offset);
                //o.uv = TRANSFORM_TEX(IN[0].uv, _MainTex);
                //o.wvel = length(IN[0].wvel);
                //triStream.Append(o);

                //o.vertex = UnityObjectToClipPos(bot + rt*0.01);
                //o.uv = TRANSFORM_TEX(IN[1].uv, _MainTex);
                //o.wvel = length(IN[0].wvel);
                //triStream.Append(o);

                //o.vertex = UnityObjectToClipPos(bot - rt*0.01);
                //o.uv = TRANSFORM_TEX(IN[2].uv, _MainTex);
                //o.wvel = length(IN[2].wvel);
                //triStream.Append(o);

                //triStream.RestartStrip();

                float triangleWidth = 0.004;

                o.tint = 0.8;
                o.vertex = UnityObjectToClipPos(bot + offset);
                o.uv = TRANSFORM_TEX(IN[0].uv, _MainTex);
                o.wvel = length(IN[0].wvel);
                
                triStream.Append(o);


                o.tint = 0.8;
                o.vertex = UnityObjectToClipPos(bot + rt * triangleWidth);
                o.uv = TRANSFORM_TEX(IN[1].uv, _MainTex);
                o.wvel = length(IN[0].wvel);
                triStream.Append(o);

                o.tint = 0.8;
                o.vertex = UnityObjectToClipPos(bot + local_up * triangleWidth);
                o.uv = TRANSFORM_TEX(IN[2].uv, _MainTex);
                o.wvel = length(IN[2].wvel);
                triStream.Append(o);



                triStream.RestartStrip();

                o.tint = 0.9;
                o.vertex = UnityObjectToClipPos(bot + offset);
                o.uv = TRANSFORM_TEX(IN[0].uv, _MainTex);
                o.wvel = length(IN[0].wvel);
                triStream.Append(o);

                o.tint = 0.9;
                o.vertex = UnityObjectToClipPos(bot - rt * triangleWidth);
                o.uv = TRANSFORM_TEX(IN[1].uv, _MainTex);
                o.wvel = length(IN[0].wvel);
                triStream.Append(o);

                o.tint = 0.9;
                o.vertex = UnityObjectToClipPos(bot + local_up * triangleWidth);
                o.uv = TRANSFORM_TEX(IN[2].uv, _MainTex);
                o.wvel = length(IN[2].wvel);
                triStream.Append(o);

                triStream.RestartStrip();





                o.tint = 0.4;
                o.vertex = UnityObjectToClipPos(bot + offset);
                o.uv = TRANSFORM_TEX(IN[0].uv, _MainTex);
                o.wvel = length(IN[0].wvel);
                triStream.Append(o);

                o.tint = 0.4;
                o.vertex = UnityObjectToClipPos(bot + rt * triangleWidth);
                o.uv = TRANSFORM_TEX(IN[1].uv, _MainTex);
                o.wvel = length(IN[0].wvel);
                triStream.Append(o);

                o.tint = 0.4;
                o.vertex = UnityObjectToClipPos(bot - local_up * triangleWidth);
                o.uv = TRANSFORM_TEX(IN[2].uv, _MainTex);
                o.wvel = length(IN[2].wvel);
                triStream.Append(o);

                triStream.RestartStrip();

                o.tint = 0.5;
                o.vertex = UnityObjectToClipPos(bot + offset);
                o.uv = TRANSFORM_TEX(IN[0].uv, _MainTex);
                o.wvel = length(IN[0].wvel);
                triStream.Append(o);

                o.tint = 0.5;
                o.vertex = UnityObjectToClipPos(bot - rt * triangleWidth);
                o.uv = TRANSFORM_TEX(IN[1].uv, _MainTex);
                o.wvel = length(IN[0].wvel);
                triStream.Append(o);

                o.tint = 0.5;
                o.vertex = UnityObjectToClipPos(bot - local_up * triangleWidth);
                o.uv = TRANSFORM_TEX(IN[2].uv, _MainTex);
                o.wvel = length(IN[2].wvel);
                triStream.Append(o);

                triStream.RestartStrip();





                //o.vertex = UnityObjectToClipPos(root_pos) + v1;
                //o.uv = TRANSFORM_TEX(IN[0].uv, _MainTex);
                //o.wvel = length(IN[0].wvel);
                //triStream.Append(o);

                //o.vertex = UnityObjectToClipPos(root_pos) + v2;
                //o.uv = TRANSFORM_TEX(IN[1].uv, _MainTex);
                //o.wvel = length(IN[0].wvel);
                //triStream.Append(o);

                //o.vertex = UnityObjectToClipPos(root_pos - IN[0].wvel * 10 * _TrailLength) + v3;
                //o.uv = TRANSFORM_TEX(IN[2].uv, _MainTex);
                //o.wvel = length(IN[0].wvel);
                //triStream.Append(o);

                //triStream.RestartStrip();



                //o.vertex = UnityObjectToClipPos(IN[0].vertex + v4);
                //o.uv = TRANSFORM_TEX(IN[0].uv, _MainTex);
                //triStream.Append(o);

                //o.vertex = UnityObjectToClipPos(IN[0].vertex + v5);
                //o.uv = TRANSFORM_TEX(IN[1].uv, _MainTex);
                //triStream.Append(o);

                //o.vertex = UnityObjectToClipPos(IN[0].vertex + v6);
                //o.uv = TRANSFORM_TEX(IN[2].uv, _MainTex);
                //triStream.Append(o);

                //triStream.RestartStrip();



                //o.vertex = UnityObjectToClipPos(IN[0].vertex + v5);
                //o.uv = TRANSFORM_TEX(IN[0].uv, _MainTex);
                //triStream.Append(o);

                //o.vertex = UnityObjectToClipPos(IN[0].vertex + v6);
                //o.uv = TRANSFORM_TEX(IN[1].uv, _MainTex);
                //triStream.Append(o);

                //o.vertex = UnityObjectToClipPos(IN[0].vertex + v7);
                //o.uv = TRANSFORM_TEX(IN[2].uv, _MainTex);
                //triStream.Append(o);

                //triStream.RestartStrip();

                /////////////////////////////////////
                //////// Break for Cohesion /////////
                /////////////////////////////////////



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

                return col*i.tint;
            }
            ENDCG
        }
    }
}
