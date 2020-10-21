Shader "SilVR/Boids/RT_sim"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
        _VelocityScale("Velocity Scale", float) = 1.0
        _TimeScale("Time Scale", float) = 1.0
        _AttractionScale("Attraction Scale", float) = 1.0
        _RepulsionScale("Repulsion Scale", float) = 1.0
        _AlignmentScale("Alignment Scale", float) = 1.0
        _GroupingScale("Grouping Scale", float) = 1.0
        _BoundaryScale("Boundary Scale", float) = 1.0
        _ClosenessScale("Closeness Scale", float) = 1.0
        _ExclusionScale("Exclusion Scale", float) = 1.0
        _SpeedLimit("Speed limit scale", float) = 1.0
        _WorldPosTarget("Target Texture", 2D) = "black" {}
        _Noise("Noise Texture", 2D) = "black" {}
        _Damping("Velocity Damping", Range(0.9,1.0) ) = 1.0
        _SpeedNosie("Speed Nosie", float) = 0.0
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
            float _AlignmentScale;
            float _GroupingScale;
            float _BoundaryScale;
            float _ClosenessScale;
            float _ExclusionScale;





            float _SpeedLimit;
            float _Damping;

            sampler2D _Noise;
            float _SpeedNoise;


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




            // Generate a discriminator. This is a 4-vector that can be used to strip out a specific index of a vector.
            // Kinda of a roundabout way to do this, but accesing vectors as arrays doesn't work unless the variable
            // we're using used the const keywork in CG/HLSL. Definitely a really hacky over complicated way of doing this
            fixed4 getDiscriminator(float2 uv)
            {
                float v = uv.y;
                int index = (int)(v * 4.0);

                // float4 representing uv.y equally, scaled between 0.0-3.9999
                float4 vec = float4(1.0, 1.0, 1.0, 1.0)*4.0*uv.y;

                // cast the vector to an integer, rescaling beween 0-3. Additionally, subtract our desired index from each one, so that if
                // the target v index matches, it will be zero in only that position
                int4 vec_i = (int4)vec - int4(0, 1, 2, 3);

                // Square the above vector component-wise to make it strictly positive. Our desired index will be exactly zero.
                // Any incorrect ones will then be greater than or equal to one.
                int4 vec_i2 = vec_i * vec_i;

                // Subtract the vector from a vector of ones, leaving our desired index as the only positive one, and leaving it equal to
                // exactly one
                fixed4 output = int4(1, 1, 1, 1) - vec_i2;

                // Saturate the output vector, setting all negative values to 0.0. Now, lets go over output
                // if uv is in first vertical block, (1 0 0 0)
                // if uv is in the second            (0 1 0 0)
                // third                             (0 0 1 0)
                // fourth                            (0 0 0 1)
                //
                // The result is a 'discriminator' vector for which we can, without using indices, strip out the desired component
                // of another vector, in form of float4 using component wise multiplication, or as a float using a dot product.
                // The purpose of this overengineered solution is to avoid using any if statements or comparisons. I'm actually
                //
                // TODO: investigate performance differences between casting as int4 and then doing calculations, then casting back
                // versus casting as int4 then back and doing calculations after
                // 
                // Additionally, if using discriminator to output directly, saturate might not be needed, in which case I can either
                // Add an additional method, or simply remove this and saturate outside of method if needed

                return saturate(output);


            }

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

            // Generate uv coordinates for a particle of a given index.
            float2 ParticleIndexToUV(int particleIndex)
            {
                int x = particleIndex % 32;
                int y = particleIndex / 32;
                float2 uv = float2(float(x) + 0.5, float(y) + 0.5) * (1.0 / 32.0);
                return uv;
            }

            // Get a struct of particle data given some uv coordinates
            ParticleData SampleParticleInfo(float2 uv)
            {
                ParticleData partData;

                // Generate base uv's for the bottom corner of the uv-grid. The 'Base UV' is used as if there was only one square
                // of pixels, being the entirety of the render plane rather than divided up into our 8 groups.


                //float2 uv = ParticleIndexToUV(index);
                float2 baseUv = generateBaseUv(uv);

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
                float4 wPosCol = tex2D(_MainTex, wPosUv);

                float4 xVelCol = tex2D(_MainTex, xVelUv);
                float4 yVelCol = tex2D(_MainTex, yVelUv);
                float4 zVelCol = tex2D(_MainTex, zVelUv);
                float4 wVelCol = tex2D(_MainTex, wVelUv);


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
                // Lets calculate the gravitational vector (in 3D). Atm, we have a world target set up with a 
                // camera, but this may change later.
                float3 gvec = normalize(point1-point2);

                // Now lets calculate the gravitational force. First we will declare some variables in case we want to make this
                // physically accurate, but we probably wont do too much with this. Possibly make the particle mass different per
                // particle to add variety, but we'll see. Actually the acceleration is independent of the mass, just realized that.
                // We'll figure something else out
                float m1 = 1.0;
                float m2 = 1.0;
                float G = 1.0;

                // Do the actual calculation for the gravitational force. For performance, we will ommit m1 as we divide by it converting
                // between force and acceleration
                float gfor = G * m2 / ( sqrt(dot(gvec, gvec)) );

                // Handle the edge case where our vector is zero, and normalization of it would result in infinities or NaN's.
                if (length(gvec) < 0.01)
                {
                    gfor = 0.0;
                }

                return gfor * gvec;
            }

            // Calculates Interboid forces, including forces of repulsion between nearby boids, the force
            // causing them to align with one another, and the force steering them towards the center of
            // the nearest handful of boids
            float3 CalculateInterboidForces(float3 pos, float3 vel, float3 noise)
            {
                // Initialize spots for the nearest particle. They aren't intended to be ordered, but they may
                // end up that way by program design.
                ParticleData closestParticle0 = SampleSerializedParticleInfo(0);
                ParticleData closestParticle1 = SampleSerializedParticleInfo(0);
                ParticleData closestParticle2 = SampleSerializedParticleInfo(0);
                ParticleData closestParticle3 = SampleSerializedParticleInfo(0);
                ParticleData closestParticle4 = SampleSerializedParticleInfo(0);
                ParticleData closestParticle5 = SampleSerializedParticleInfo(0);
                ParticleData closestParticle6 = SampleSerializedParticleInfo(0);
                ParticleData closestParticle7 = SampleSerializedParticleInfo(0);

                // initialize an empty spot for our net force
                float3 netForce = float3(0.0, 0.0, 0.0);

                // Go through all 1024 boids. We will denote the boid we are calculating the force for as our boid, and the boids we are 
                // iterating through as particles for ease of reference.
                for (int i = 0; i < 1024; i++)
                {


                    // Sample our nth particle
                    ParticleData particle = SampleSerializedParticleInfo(i);
                    // If the sampled partlicle is in front of our boid and closer than the current closest
                    if (dot(vel, particle.pos - pos) > 0.0 && distance(particle.pos, pos) < distance(pos, closestParticle0.pos))
                    {
                        // Update the nearest particle list.
                        closestParticle7 = closestParticle6;
                        closestParticle6 = closestParticle5;
                        closestParticle5 = closestParticle4;
                        closestParticle4 = closestParticle3;
                        closestParticle3 = closestParticle2;
                        closestParticle2 = closestParticle1;
                        closestParticle1 = closestParticle0;
                        closestParticle0 = particle;
                    }
                 
                    // Initialize some epsilon.
                    float EPSILON = 0.000001;
                        
                    // Note we are currently in a loop that goes through all the particles. So for the current particle, calculate the offset between
                    // the current boid and the sampled particle in question.
                    float3 offset = pos - particle.pos.xyz;

                    

                    // equally blend between 4 of the 7 nearest particles. Kind of inconsistent, but changing this would alter the tuning of the boids
                    // so we're gonna leave this in too. This should actually be outside of the for loop, but uh, we'll leave it in here because its working.
                    // This is entirely incorrect though, so oof.
                    float3 newVelocity = lerp(lerp(closestParticle0.vel, closestParticle1.vel, 0.5), lerp(closestParticle2.vel, closestParticle3.vel, 0.5), 0.5);
                    
                    // The excluded variable determines whether or not to steer towards or away the nearest boids direction (well, 4 of the nearest boids
                    // directions.) The two cases we want to steer away is when the distance to the nearest particle is either below some epsilon
                    // close to zero, or if the distance is less than a user specified distance.
                    float steerTowardsMultiplier = 1.0;

                    // Calculate some weighted average of the nearest particles. We give the actual closest particle more weight kinda for no reason,
                    // but note that the next 7 closest particles may not neccesarily be in order, so giving the closest particle the most weight or
                    // giving them all equal weight are the only two consistent options. This is another thing that should technically be outside the for
                    // loop, but for now we'll leave it as the boids are tuned pretty well to my preference.
                    float3 centerMass = (closestParticle0.pos * (9.0 / 16.0) +
                                         closestParticle1.pos * (1.0 / 16.0) +
                                         closestParticle2.pos * (1.0 / 16.0) +
                                         closestParticle3.pos * (1.0 / 16.0) +
                                         closestParticle4.pos * (1.0 / 16.0) +
                                         closestParticle5.pos * (1.0 / 16.0) +
                                         closestParticle6.pos * (1.0 / 16.0) +
                                         closestParticle7.pos * (1.0 / 16.0) );
                    
                    // If our current particle in question is within 5 meters away, but further away than some epsilon (to prevent
                    // divisions by zero
                    if (length(offset) < 5.0 && length(offset) > EPSILON)
                    {
                        // average the two particles velocites together
                       float3 average_vel = 0.5 * (particle.vel.xyz + vel.xyz);
                        
                        // calculate the difference in the velocies
                        float3 offset_velocity = particle.vel - average_vel;
                        
                        // Add our repulsional force. Not sure why this appears positive, but uh, its working. I think calculate
                        // gravitational force is incorrect naming in the sense that its simply an inverse square repulsion
                        netForce += CalculateInverseSquareRepulsion(pos.xyz, particle.pos.xyz) * _RepulsionScale * 0.0001;
                      
                        // If the closest particle is closer than some user specified range
                        if (length(pos.xyz - closestParticle0.pos) < _ExclusionScale)
                        {
                            // exclude that particle from a future calculation
                            steerTowardsMultiplier = -1.0;
                        }
                    }

                    // If instead the closest particle is less than our epsilon
                    else if (length(offset) < EPSILON)
                    {
                        // Calculate the repulsional force as the force between our boid in question, and some hypothetical
                        // particle in a position defined by incoming color noise. This is specifically to address the issue
                        // that particles like to layer on top of one another, so I wanted to find a unique vector for each
                        // to follow under this case, which comes from the same thing that colors the boids.
                        netForce += CalculateInverseSquareRepulsion(pos.xyz, pos.xyz + noise*0.0001) * _ClosenessScale * 0.0001;
                        
                        // Also exclude the nearest particle from a specific calculation
                        steerTowardsMultiplier = -1.0;
                    }


                    // If we aren't already in the weighted average of the nearest boids' positions
                    if (length(centerMass - pos.xyz) > 0.00001)
                    {
                        // Pull ourselves towards the weighted average of the nearest boids' positions
                        netForce -= CalculateInverseSquareRepulsion(pos.xyz, centerMass) * _GroupingScale * 0.0001;
                    }

                    // if the closest particle is within 10 meters
                    if (length(closestParticle0.pos - pos.xyz) < 10)
                    {
                        // Attempt to align with the nearest particles.
                        newVelocity = lerp(vel, newVelocity, _AlignmentScale);
                        
                        // align with the nearest particles, unless excluded has been set to -1, in which the particle will attempt to turn away from the
                        // particle.
                        netForce -= (vel - newVelocity)*steerTowardsMultiplier;
                    }
                }

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
                // Get the information about the world target. This is mostly left over from when these boids
                // were generic gravity gpu particles, but it helps the boundary repulsion keep the boids towards the center
                // so its still in use. The user can set attraction scale property to zero on the sim material to disable this.
                float4 wt_x = tex2D(_WorldPosTarget, float2(0.0, 0.0) + float2(0.25, 0.25));
                float4 wt_y = tex2D(_WorldPosTarget, float2(0.5, 0.0) + float2(0.25, 0.25));
                float4 wt_z = tex2D(_WorldPosTarget, float2(0.0, 0.5) + float2(0.25, 0.25));
                float4 wt_w = tex2D(_WorldPosTarget, float2(0.5, 0.5) + float2(0.25, 0.25));

                float wt_xf = RGBA8ToFloat32(wt_x);
                float wt_yf = RGBA8ToFloat32(wt_y);
                float wt_zf = RGBA8ToFloat32(wt_z);
                float wt_wf = RGBA8ToFloat32(wt_w);

                // Generate the world targets position as a 3-vector.
                float3 worldTarget = float3(wt_xf, wt_yf, wt_zf);

                // Generate some discriminator based on the uv. This creates a float4 that strips
                // out all but some desired index of information from a value. That info is determined
                // by the range of the v coordinate passed in.
                float4 discriminator = getDiscriminator(i.uv);

                // too lazy to specify our v2f as the object owning the uv, so lets declare a new float2.
                // This'll be compiled out anyway.
                float2 uv = i.uv;

                // Sample our current boid's info based on the incoming uv coordinate.
                ParticleData particle = SampleParticleInfo(uv);

                // Now lets construct the position and velocity vectors
                float4 particle_pos = particle.pos;
                float4 particle_vel = particle.vel;

                // Now lets update the particles' positions based on velocity. First, scale this down to match the target frame rate, then
                // scale it back up according to user specification


                ////////////////////////////////////////////////////////
                ////              Particle Simulation               ////
                ////////////////////////////////////////////////////////

                // sample some noise coloe
                float4 noise_color = tex2D(_Noise, i.uv).x;

                // Lets start with a base time scale, specified by the user, divided to match the target frame rate
                float time_scale = _TimeScale * (1.0 / 90.0);

                // Calculate all the forces acting on our boid.
                float3 gfor = CalculateInverseSquareRepulsion(worldTarget.xyz, particle_pos.xyz);
                float3 rfor = CalculateInterboidForces(particle_pos.xyz, particle_vel.xyz, noise_color.xyz);
                float3 bfor = CalculateBoundaryRepulsion(particle_pos.xyz, 10.0);


                // Generate the acceleration and velocity vectors. Gfor and bfor are adjustable by parameters. rfor was the repulsive force, but is now any
                // interboid force, and because any interboid force requires iterating through all boids, they were placed in the same method for optimzation
                //purposes. they are adjusted via parameter inside the method.

                // TODO: Make parameter adjustments consistent, either by moving the parameters to the interior of the gfor and bfor calculation methods,
                // or return the interboid forces as a struct. Basically, anything to make this more consistent.
                float3 a_vec = gfor * _AttractionScale + rfor  + bfor * _BoundaryScale;

                // Get our boid's previous velocity
                float3 v_vec = particle_vel;

                // update our position and velocities.
                particle_pos += float4(v_vec, 0.0) * _TimeScale * (1.00+0.15*noise_color.x);
                particle_vel += float4(a_vec, 0.0) * _TimeScale * (1.00+0.15*noise_color.x);
                
                // Clamp our velocity to be within a certain length. This is mainly because the velocity kept reaching zero
                // (resulting in divisions by zero in normalizing vectors of length 0) or exploding off towards infinity.
                // Really, we want our boids going approximately the same velocity as well.
                particle_vel = (clamp(length(particle_vel), 0.05, 0.1))  * normalize(particle_vel) + noise_color * _SpeedNoise;



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