Shader "SilVR/Boids/Noise Processor"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
        _WorldTarget("Target", 2D) = "black" {}
        _WorldPos("World Pos", Vector) = (0,0,0,0)
        _WorldVel("World Vel", Vector) = (0,0,0,0)
        _Noise("Noise Texture", 2D) = "black" {}
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
            #include "ShaderUtils.cginc"

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float2 uv : TEXCOORD0;
                float4 vertex : SV_POSITION;
                float4 worldPos : TEXCOORD1;
            };

            sampler2D _MainTex;
            sampler2D _Noise;
            float4 _MainTex_ST;
            float4 _WorldPos;
            float4 _WorldVel;

            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = TRANSFORM_TEX(v.uv, _MainTex);

                float4 baseUv = float4(o.uv, 0, 0);
                float4 dataIndex = float4(0.0, 0.25, 0.0, 0.0);
                float4 posVelIndex = float4(0.5, 0.0, 0.0, 0.0);

                float4 xPosUv = baseUv + 0.0 * dataIndex + 0.0 * posVelIndex;
                float4 yPosUv = baseUv + 1.0 * dataIndex + 0.0 * posVelIndex;
                float4 zPosUv = baseUv + 2.0 * dataIndex + 0.0 * posVelIndex;
                float4 wPosUv = baseUv + 3.0 * dataIndex + 0.0 * posVelIndex;

                float xPos = RGBA8ToFloat32(tex2Dlod(_MainTex, xPosUv));
                float yPos = RGBA8ToFloat32(tex2Dlod(_MainTex, yPosUv));
                float zPos = RGBA8ToFloat32(tex2Dlod(_MainTex, zPosUv));
                float wPos = RGBA8ToFloat32(tex2Dlod(_MainTex, wPosUv));
                
                float4 worldPos = float4(xPos, yPos, zPos, wPos);

                o.worldPos = worldPos;

                return o;
            }

            fixed4 frag(v2f i) : SV_Target
            {
                // Declare some empty floats so we can convert them to colors later
                float pos;
                float vel;


                float4 baseUv = float4(i.uv * float2(0.5, 0.25), 0, 0);
                float4 dataIndex = float4(0.0, 0.25, 0.0, 0.0);
                float4 posVelIndex = float4(0.5, 0.0, 0.0, 0.0);

                float2 xPosUv = float2(0.0, 0.0) + float2(0.25, 0.25);
                float2 yPosUv = float2(0.5, 0.0) + float2(0.25, 0.25);
                float2 zPosUv = float2(0.0, 0.5) + float2(0.25, 0.25);
                float2 wPosUv = float2(0.5, 0.5) + float2(0.25, 0.25);

                float xPos = RGBA8ToFloat32(tex2D(_MainTex, xPosUv));
                float yPos = RGBA8ToFloat32(tex2D(_MainTex, yPosUv));
                float zPos = RGBA8ToFloat32(tex2D(_MainTex, zPosUv));
                float wPos = RGBA8ToFloat32(tex2D(_MainTex, wPosUv));

                float4 cWorldPos = float4(xPos, yPos, zPos, 0.0);


                // Take in some noise texture and generate a vector from it
                float3 noiseVector = tex2D(_Noise, i.uv).xyz;

                // Get the world position specified by user and add a noise vector to it
                float4 worldPos = float4(noiseVector - float3(0.5, 0.5, 0.5), 0.0);

                // Control the index of the incoming vectors based on the uv coordinates in the same manner the
                // gpu particles use
                if (0.0 < i.uv.y && i.uv.y < 0.25)
                {
                    pos = worldPos.x;
                    vel = _WorldVel.x;
                }
                if (0.25 < i.uv.y && i.uv.y < 0.50)
                {
                    pos = worldPos.y;
                    vel = _WorldVel.y;

                }
                if (0.50 < i.uv.y && i.uv.y < 0.75)
                {
                    pos = worldPos.z;
                    vel = _WorldVel.z;

                }
                if (0.75 < i.uv.y && i.uv.y < 1.00)
                {
                    pos = worldPos.w;
                    vel = _WorldVel.w;
                }

                // Use the horizontal uv to determine whether or not we are outputting position or velocity
                float val;
                if (i.uv.x > 0.5)
                {
                    val = float4(1, 0, 0, 0);
                }
                else
                {
                    val = pos;
                }

                // Convert the determined value to an RGBA8 to return to the fragment shader
                float4 col = float32ToRGBA8(val);

                // Return our calculated color
                return col;
            }
            ENDCG
        }
    }
}
