/**
 * FWaveSolver.hpp
 *
 ****
 **** F-Wave Riemann Solver for the Shallow Water Equation
 ****
 *
 *  Created on: Aug 25, 2011
 *  Last Update: Feb 18, 2012
 *
 ****
 *
 *  Author: Alexander Breuer
 *    Homepage: http://www5.in.tum.de/wiki/index.php/Dipl.-Math._Alexander_Breuer
 *    E-Mail: breuera AT in.tum.de
 *
 ****
 *
 * (Main) Literature:
 *
 *   @article{bale2002wave,
 *            title={A wave propagation method for conservation laws and balance laws with spatially varying flux
 *functions}, author={Bale, D.S. and LeVeque, R.J. and Mitran, S. and Rossmanith, J.A.}, journal={SIAM Journal on
 *Scientific Computing}, volume={24}, number={3}, pages={955--978}, year={2002}, publisher={Citeseer}}
 *
 *   @book{leveque2002finite,
 *         Author = {LeVeque, R. J.},
 *         Date-Added = {2011-09-13 14:09:31 +0000},
 *         Date-Modified = {2011-10-31 09:46:40 +0000},
 *         Publisher = {Cambridge University Press},
 *         Title = {Finite Volume Methods for Hyperbolic Problems},
 *         Volume = {31},
 *         Year = {2002}}
 *
 *   @webpage{levequeclawpack,
 *            Author = {LeVeque, R. J.},
 *            Lastchecked = {January, 05, 2011},
 *            Title = {Clawpack Sofware},
 *            Url = {https://github.com/clawpack/clawpack-4.x/blob/master/geoclaw/2d/lib}}
 *
 ****
 *
 * Acknowledgments:
 *   Special thanks go to R.J. LeVeque and D.L. George for publishing their code
 *   and the corresponding documentation (-> Literature).
 */

#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <immintrin.h>
#include "WavePropagationSolver.hpp"



namespace Solvers {

  /**
   * FWave Riemann Solver for the Shallow Water Equations.
   *
   * T should be double or float.
   */
  template <class T>
  class FWaveSolver: public WavePropagationSolver<float> {
  private:
    // Use nondependent names (template base class)
    using WavePropagationSolver<float>::dryTol_;
    using WavePropagationSolver<float>::gravity_;
    using WavePropagationSolver<float>::zeroTol_;
    using WavePropagationSolver<float>::hLeft_;
    using WavePropagationSolver<float>::hRight_;
    using WavePropagationSolver<float>::huLeft_;
    using WavePropagationSolver<float>::huRight_;
    using WavePropagationSolver<float>::bLeft_;
    using WavePropagationSolver<float>::bRight_;
    using WavePropagationSolver<float>::uLeft_;
    using WavePropagationSolver<float>::uRight_;

    using WavePropagationSolver<float>::wetDryState_;

    /**
     * Compute the edge local eigenvalues.
     *
     * @param o_waveSpeeds will be set to: speeds of the linearized waves (eigenvalues).
     */
    void computeWaveSpeeds(__m256 o_waveSpeeds[2]) const {
      // Compute eigenvalues of the Jacobian matrices in states Q_{i-1} and Q_{i}
      __m256 characteristicSpeeds[2]{};

      characteristicSpeeds[0] = _mm256_sub_ps(uLeft_, _mm256_sqrt_ps(_mm256_mul_ps(_mm256_set1_ps(gravity_), hLeft_)));
      characteristicSpeeds[1] = _mm256_add_ps(uRight_, _mm256_sqrt_ps(_mm256_mul_ps(_mm256_set1_ps(gravity_), hRight_)));

      // Compute "Roe speeds"
      
      __m256 hRoe = _mm256_mul_ps(_mm256_set1_ps(0.5), _mm256_add_ps(hRight_, hLeft_));

      __m256 sqrtHL = _mm256_sqrt_ps(hLeft_);
      __m256 sqrtHR = _mm256_sqrt_ps(hRight_);

      __m256 uRoe = _mm256_add_ps(_mm256_mul_ps(uLeft_, sqrtHL), _mm256_mul_ps(uRight_, sqrtHR));

      uRoe = _mm256_div_ps(uRoe, _mm256_add_ps(sqrtHL, sqrtHR));

      __m256 roeSpeeds[2]{};

      roeSpeeds[0] = _mm256_sub_ps(uRoe, _mm256_sqrt_ps(_mm256_mul_ps(_mm256_set1_ps(gravity_), hRoe)));
      roeSpeeds[1] = _mm256_add_ps(uRoe, _mm256_sqrt_ps(_mm256_mul_ps(_mm256_set1_ps(gravity_), hRoe)));

      // Compute eindfeldt speeds
      __m256 einfeldtSpeeds[2]{};
      einfeldtSpeeds[0] = _mm256_min_ps(characteristicSpeeds[0], roeSpeeds[0]);
      einfeldtSpeeds[1] = _mm256_max_ps(characteristicSpeeds[1], roeSpeeds[1]);

      // Set wave speeds
      o_waveSpeeds[0] =einfeldtSpeeds[0];
      o_waveSpeeds[1] = einfeldtSpeeds[1];
    }

    /**
     * Compute the decomposition into f-Waves.
     *
     * @param waveSpeeds speeds of the linearized waves (eigenvalues).
     * @param o_fWaves will be set to: Decomposition into f-Waves.
     */
    void computeWaveDecomposition(const __m256 waveSpeeds[2], __m256 o_fWaves[2][2]) const {
      // Eigenvalues***********************************************************************************************
      // Computed somewhere before.
      // An option would be to use the char. Speeds:
      //
      // lambda^1 = u_{i-1} - sqrt(g * h_{i-1})
      // lambda^2 = u_i     + sqrt(g * h_i)
      // Matrix of right eigenvectors******************************************************************************
      //     1                              1
      // R =
      //     u_{i-1} - sqrt(g * h_{i-1})    u_i + sqrt(g * h_i)
      // **********************************************************************************************************
      //                                                                      u_i + sqrt(g * h_i)              -1
      // R^{-1} = 1 / (u_i - sqrt(g * h_i) - u_{i-1} + sqrt(g * h_{i-1}) *
      //                                                                   -( u_{i-1} - sqrt(g * h_{i-1}) )     1
      // **********************************************************************************************************
      //                hu
      // f(q) =
      //         hu^2 + 1/2 g * h^2
      // **********************************************************************************************************
      //                                    0
      // \delta x \Psi =
      //                  -g * 1/2 * (h_i + h_{i-1}) * (b_i - b_{i+1})
      // **********************************************************************************************************
      // beta = R^{-1} * (f(Q_i) - f(Q_{i-1}) - \delta x \Psi)
      // **********************************************************************************************************

      // assert: wave speed of the 1st wave family should be less than the speed of the 2nd wave family.
      //assert(waveSpeeds[0] < waveSpeeds[1]);

      __m256 lambdaDif = _mm256_sub_ps(waveSpeeds[1], waveSpeeds[0]);

      // assert: no division by zero
      //assert(std::abs(lambdaDif) > zeroTol_);

      // Compute the inverse matrix R^{-1}
      __m256 Rinv[2][2]{};

      __m256 oneDivLambdaDif = _mm256_div_ps(_mm256_set1_ps(1.0), lambdaDif);
      Rinv[0][0]        = _mm256_mul_ps(oneDivLambdaDif, waveSpeeds[1]);
      Rinv[0][1]        = _mm256_mul_ps(_mm256_set1_ps(-1.0),oneDivLambdaDif);

      Rinv[1][0] = _mm256_mul_ps(oneDivLambdaDif, _mm256_mul_ps(_mm256_set1_ps(-1.0),waveSpeeds[0]));
      Rinv[1][1] = oneDivLambdaDif;

      // Right hand side
      __m256 fDif[2]{};

      // Calculate modified (bathymetry!) flux difference_mm256_set1_pd(0.5)
      // f(Q_i) - f(Q_{i-1})
      fDif[0] = _mm256_sub_ps(huRight_, huLeft_);
      fDif[1] = _mm256_sub_ps(_mm256_add_ps(_mm256_mul_ps(huRight_,uRight_), _mm256_mul_ps(_mm256_mul_ps(_mm256_mul_ps(_mm256_set1_ps(0.5), _mm256_set1_ps(gravity_)), hRight_), hRight_))
                , _mm256_add_ps(_mm256_mul_ps(huLeft_, uLeft_), _mm256_mul_ps(_mm256_mul_ps(_mm256_mul_ps(_mm256_set1_ps(0.5), _mm256_set1_ps(gravity_)), hLeft_), hLeft_)));

      // \delta x \Psi[2]
      __m256 psi = _mm256_mul_ps(_mm256_mul_ps(_mm256_mul_ps(_mm256_set1_ps(-gravity_), _mm256_set1_ps(0.5)), _mm256_add_ps(hRight_, hLeft_)), _mm256_sub_ps(bRight_, bLeft_));
      fDif[1] = _mm256_sub_ps(fDif[1],psi);

      // Solve linear equations
      __m256 beta[2]{};
      beta[0] = _mm256_add_ps(_mm256_mul_ps(Rinv[0][0], fDif[0]), _mm256_mul_ps(Rinv[0][1], fDif[1]));
      beta[1] = _mm256_add_ps(_mm256_mul_ps(Rinv[1][0], fDif[0]), _mm256_mul_ps(Rinv[1][1], fDif[1]));

      // Return f-waves
      o_fWaves[0][0] = beta[0];
      o_fWaves[0][1] = _mm256_mul_ps(beta[0], waveSpeeds[0]);

      o_fWaves[1][0] = beta[1];
      o_fWaves[1][1] = _mm256_mul_ps(beta[1], waveSpeeds[1]);
    }

    /**
     * Compute net updates for the cell on the left/right side of the edge.
     * Its assumed that the member variables are set already.
     *
     * @param waveSpeeds speeds of the linearized waves (eigenvalues).
     *
     * @param o_hUpdateLeft will be set to: Net-update for the height of the cell on the left side of the edge.
     * @param o_hUpdateRight will be set to: Net-update for the height of the cell on the right side of the edge.
     * @param o_huUpdateLeft will be set to: Net-update for the momentum of the cell on the left side of the edge.
     * @param o_huUpdateRight will be set to: Net-update for the momentum of the cell on the right side of the edge.
     * @param o_maxWaveSpeed will be set to: Maximum (linearized) wave speed -> Should be used in the CFL-condition.
     */
    void computeNetUpdatesWithWaveSpeeds(
      const __m256 waveSpeeds[2],
      __m256&      o_hUpdateLeft,
      __m256&      o_hUpdateRight,
      __m256&      o_huUpdateLeft,
      __m256&      o_huUpdateRight,
      __m256&      o_maxWaveSpeed
    ) {
      // Reset net updates
      o_hUpdateLeft = o_hUpdateRight = o_huUpdateLeft = o_huUpdateRight = _mm256_setzero_ps();

      //! Where to store the two f-waves
      __m256 fWaves[2][2];

      // Compute the decomposition into f-waves
      computeWaveDecomposition(waveSpeeds, fWaves);

      // Compute the net-updates
      // 1st wave family

      float waveSpeeds04Arr[8];_mm256_storeu_ps(waveSpeeds04Arr, waveSpeeds[0]);
      float waveSpeeds14Arr[8];_mm256_storeu_ps(waveSpeeds14Arr, waveSpeeds[1]);
      float fWaves004Arr[8];_mm256_storeu_ps(fWaves004Arr, fWaves[0][0]);
      float fWaves014Arr[8];_mm256_storeu_ps(fWaves014Arr, fWaves[0][1]);
      float fWaves104Arr[8];_mm256_storeu_ps(fWaves104Arr, fWaves[1][0]);
      float fWaves114Arr[8];_mm256_storeu_ps(fWaves114Arr, fWaves[1][1]);
      float o_hUpdateLeft4Arr[8];_mm256_storeu_ps(o_hUpdateLeft4Arr, o_hUpdateLeft);
      float o_huUpdateLeft4Arr[8];_mm256_storeu_ps(o_huUpdateLeft4Arr, o_huUpdateLeft);
      float o_hUpdateRight4Arr[8];_mm256_storeu_ps(o_hUpdateRight4Arr, o_hUpdateRight);
      float o_huUpdateRight4Arr[8];_mm256_storeu_ps(o_huUpdateRight4Arr, o_huUpdateRight);
      for(int i=0;i<8;i++){
          if (waveSpeeds04Arr[i] < -zeroTol_) { // Left going
            o_hUpdateLeft4Arr[i] += fWaves004Arr[i];
            o_huUpdateLeft4Arr[i] += fWaves014Arr[i];
          } else if (waveSpeeds04Arr[i] > zeroTol_) { // Right going
            o_hUpdateRight4Arr[i] += fWaves004Arr[i];
            o_huUpdateRight4Arr[i] += fWaves014Arr[i];
          } else { // Split waves
            o_hUpdateLeft4Arr[i] += (0.5) * fWaves004Arr[i];
            o_huUpdateLeft4Arr[i] += (0.5) * fWaves014Arr[i];
            o_hUpdateRight4Arr[i] += (0.5) * fWaves004Arr[i];
            o_huUpdateRight4Arr[i] += (0.5) * fWaves014Arr[i];
          }

          // 2nd wave family
          if (waveSpeeds14Arr[i] < -zeroTol_) { // Left going
            o_hUpdateLeft4Arr[i] += fWaves104Arr[i];
            o_huUpdateLeft4Arr[i] += fWaves114Arr[i];
          } else if (waveSpeeds14Arr[i] > zeroTol_) { // Right going
            o_hUpdateRight4Arr[i] += fWaves104Arr[i];
            o_huUpdateRight4Arr[i] += fWaves114Arr[i];
          } else { // Split waves
            o_hUpdateLeft4Arr[i] += (0.5) * fWaves104Arr[i];
            o_huUpdateLeft4Arr[i] += (0.5) * fWaves114Arr[i];
            o_hUpdateRight4Arr[i] += (0.5) * fWaves104Arr[i];
            o_huUpdateRight4Arr[i] += (0.5) * fWaves114Arr[i];
          }

      }
      o_hUpdateLeft = _mm256_loadu_ps(o_hUpdateLeft4Arr);
      o_huUpdateLeft = _mm256_loadu_ps(o_huUpdateLeft4Arr);
      o_hUpdateRight = _mm256_loadu_ps(o_hUpdateRight4Arr);
      o_huUpdateRight = _mm256_loadu_ps(o_huUpdateRight4Arr);
      // Compute maximum wave speed (-> CFL-condition)
      o_maxWaveSpeed = _mm256_max_ps(_mm256_andnot_ps(_mm256_set1_ps(-0.0), waveSpeeds[0]), _mm256_andnot_ps(_mm256_set1_ps(-0.0), waveSpeeds[1]));
    }

  protected:
    /**
     * Determine the wet/dry state and set member variables accordingly.
     */
    void determineWetDryState() override {
      // Determine the wet/dry state
       float hLeft4Arr[8];_mm256_storeu_ps(hLeft4Arr, hLeft_);
       float hRight4Arr[8];_mm256_storeu_ps(hRight4Arr, hRight_);
       float uRight4Arr[8];_mm256_storeu_ps(uRight4Arr, uRight_);
       float huRight4Arr[8];_mm256_storeu_ps(huRight4Arr, huRight_);
       float bLeft4Arr[8];_mm256_storeu_ps(bLeft4Arr, bLeft_);
       float bRight4Arr[8];_mm256_storeu_ps(bRight4Arr, bRight_);
       float huLeft4Arr[8];_mm256_storeu_ps(huLeft4Arr, huLeft_);
       float uLeft4Arr[8];_mm256_storeu_ps(uLeft4Arr, uLeft_);
      for(int i=0;i<8;i++){
        if (hLeft4Arr[i] < dryTol_ && hRight4Arr[i] < dryTol_) { // Both cells are dry
          wetDryState_[i] = WavePropagationSolver<float>::WetDryState::DryDry;
        } else if (hLeft4Arr[i] < dryTol_) { // Left cell dry, right cell wet
          uRight4Arr[i] = huRight4Arr[i] / hRight4Arr[i];

          // Set wall boundary conditions.
          // This is not correct in the case of inundation problems.
          hLeft4Arr[i]       = hRight4Arr[i];
          bLeft4Arr[i]       = bRight4Arr[i];
          huLeft4Arr[i]      = -huRight4Arr[i];
          uLeft4Arr[i]       = -uRight4Arr[i];
          wetDryState_[i] = WavePropagationSolver<float>::WetDryState::DryWetWall;
        } else if (hRight4Arr[i] < dryTol_) { // Left cell wet, right cell dry
          uLeft4Arr[i] = huLeft4Arr[i] / hLeft4Arr[i];

          // Set wall boundary conditions.
          // This is not correct in the case of inundation problems.
          hRight4Arr[i]      = hLeft4Arr[i];
          bRight4Arr[i]      = bLeft4Arr[i] ;
          huRight4Arr[i]     = -huLeft4Arr[i];
          uLeft4Arr[i]       = -uRight4Arr[i];
          wetDryState_[i] = WavePropagationSolver<float>::WetDryState::WetDryWall;
        } else { // Both cells wet
          uLeft4Arr[i]  = huLeft4Arr[i] / hLeft4Arr[i];
          uRight4Arr[i] = huRight4Arr[i] / hRight4Arr[i];

          wetDryState_[i] = WavePropagationSolver<float>::WetDryState::WetWet;
        }
      }
      hLeft_ = _mm256_loadu_ps(hLeft4Arr);
      hRight_ = _mm256_loadu_ps(hRight4Arr);
      uRight_ = _mm256_loadu_ps(uRight4Arr);
      huRight_ = _mm256_loadu_ps(huRight4Arr);
      bLeft_ = _mm256_loadu_ps(bLeft4Arr);
      bRight_ = _mm256_loadu_ps(bRight4Arr);
      huLeft_ = _mm256_loadu_ps(huLeft4Arr);
      uLeft_ = _mm256_loadu_ps(uLeft4Arr);
      
    }

  public:
    /**
     * Constructor of the f-Wave solver with optional parameters.
     *
     * @param dryTolerance numerical definition of "dry".
     * @param gravity gravity constant.
     * @param zeroTolerance numerical definition of zero.
     */
    FWaveSolver(
      float dryTolerance  = static_cast<float>(0.01),
      float gravity       = static_cast<float>(9.81),
      float zeroTolerance = static_cast<float>(0.000000001)
    ):
      WavePropagationSolver<float>(dryTolerance, gravity, zeroTolerance) {}

    ~FWaveSolver() override = default;

    /**
     * Compute net updates for the cell on the left/right side of the edge.
     * This is the default method of a standalone f-Wave solver.
     *
     * Please note:
     *   In the case of a Dry/Wet- or Wet/Dry-boundary, wall boundary conditions will be set.
     *   The f-Wave solver is not positivity preserving.
     *   -> You as the programmer have to take care about "negative water heights"!
     *
     * @param hLeft height on the left side of the edge.
     * @param hRight height on the right side of the edge.
     * @param huLeft momentum on the left side of the edge.
     * @param huRight momentum on the right side of the edge.
     * @param bLeft bathymetry on the left side of the edge.
     * @param bRight bathymetry on the right side of the edge.
     *
     * @param o_hUpdateLeft will be set to: Net-update for the height of the cell on the left side of the edge.
     * @param o_hUpdateRight will be set to: Net-update for the height of the cell on the right side of the edge.
     * @param o_huUpdateLeft will be set to: Net-update for the momentum of the cell on the left side of the edge.
     * @param o_huUpdateRight will be set to: Net-update for the momentum of the cell on the right side of the edge.
     * @param o_maxWaveSpeed will be set to: Maximum (linearized) wave speed -> Should be used in the CFL-condition.
     */
    void computeNetUpdates( ///////////////////////////////////////////////////////////////////////7/77/////////////////////////77
      const __m256& hLeft,
      const __m256& hRight,
      const __m256& huLeft,
      const __m256& huRight,
      const __m256& bLeft,
      const __m256& bRight,
      __m256&       o_hUpdateLeft,
      __m256&       o_hUpdateRight,
      __m256&       o_huUpdateLeft,
      __m256&       o_huUpdateRight,
      __m256&       o_maxWaveSpeed
    ) override {
      // Set speeds to zero (will be determined later)
     
      uLeft_ = uRight_ = _mm256_setzero_ps();

      // Reset the maximum wave speed
      o_maxWaveSpeed = _mm256_setzero_ps();

      //! Wave speeds of the f-wavesalignas(32) 
      __m256 waveSpeeds[2] = {_mm256_setzero_ps(),_mm256_setzero_ps()};

      // Store parameters to member variables
      WavePropagationSolver<float>::storeParameters(hLeft, hRight, huLeft, huRight, bLeft, bRight);

      // Determine the wet/dry state and compute local variables correspondingly
      determineWetDryState();
      
      // Compute the wave speeds
      computeWaveSpeeds(waveSpeeds);
      // Use the wave speeds to compute the net-updates
      computeNetUpdatesWithWaveSpeeds(
        waveSpeeds, o_hUpdateLeft, o_hUpdateRight, o_huUpdateLeft, o_huUpdateRight, o_maxWaveSpeed
      );
      
      // Zero ghost updates (wall boundary)
      float o_hUpdateLeft4Arr[8];_mm256_storeu_ps(o_hUpdateLeft4Arr, o_hUpdateLeft);
      float o_huUpdateLeft4Arr[8];_mm256_storeu_ps(o_huUpdateLeft4Arr, o_huUpdateLeft);
      float o_hUpdateRight4Arr[8];_mm256_storeu_ps(o_hUpdateRight4Arr, o_hUpdateRight);
      float o_huUpdateRight4Arr[8];_mm256_storeu_ps(o_huUpdateRight4Arr, o_huUpdateRight);
      
      for(int i=0;i<8;i++){
        if (wetDryState_[i] == WavePropagationSolver<float>::WetDryState::WetDryWall) {
          o_hUpdateRight4Arr[i]  = 0;
          o_huUpdateRight4Arr[i] = 0;
        } else if (wetDryState_[i] == WavePropagationSolver<float>::WetDryState::DryWetWall) {
          o_hUpdateLeft4Arr[i]  = 0;
          o_huUpdateLeft4Arr[i] = 0;
        }

        // Zero updates and return in the case of dry cells
        if (wetDryState_[i] == WavePropagationSolver<float>::WetDryState::DryDry) {
          o_hUpdateLeft4Arr[i] = o_hUpdateRight4Arr[i] = o_huUpdateLeft4Arr[i] = o_huUpdateRight4Arr[i] = 0.0;
        }
        //std::cout<<wetDryState_[i]<<' '<<o_hUpdateLeft4Arr[i]<<' '<<o_hUpdateRight4Arr[i]<<' '<<o_huUpdateLeft4Arr[i]<<' '<<o_huUpdateRight4Arr[i]<<std::endl;
      }
      o_hUpdateLeft = _mm256_loadu_ps(o_hUpdateLeft4Arr);
      o_huUpdateLeft = _mm256_loadu_ps(o_huUpdateLeft4Arr);
      o_hUpdateRight = _mm256_loadu_ps(o_hUpdateRight4Arr);
      o_huUpdateRight = _mm256_loadu_ps(o_huUpdateRight4Arr);

      
    }
  

  

    /**
     * Compute net updates for the cell on the left/right side of the edge.
     * This is an expert method, because a lot of (numerical-)knowledge about the problem is assumed/has to be provided.
     * It is the f-Wave entry point for the hybrid solver,  which combines the "simple" F-Wave approach with the more
     * complex Augmented Riemann Solver.
     *
     * wetDryState is assumed to be WetWet.
     *
     * @param hLeft height on the left side of the edge.
     * @param hRight height on the right side of the edge.
     * @param huLeft momentum on the left side of the edge.
     * @param huRight momentum on the right side of the edge.
     * @param bLeft bathymetry on the left side of the edge.
     * @param bRight bathymetry on the right side of the edge.
     * @param uLeft velocity on the left side of the edge.
     * @param uRight velocity on the right side of the edge.
     * @param waveSpeeds speeds of the linearized waves (eigenvalues).
     *                   A hybrid solver will typically provide its own values.
     *
     * @param o_hUpdateLeft will be set to: Net-update for the height of the cell on the left side of the edge.
     * @param o_hUpdateRight will be set to: Net-update for the height of the cell on the right side of the edge.
     * @param o_huUpdateLeft will be set to: Net-update for the momentum of the cell on the left side of the edge.
     * @param o_huUpdateRight will be set to: Net-update for the momentum of the cell on the right side of the edge.
     * @param o_maxWaveSpeed will be set to: Maximum (linearized) wave speed -> Should be used in the CFL-condition.
     */
    void computeNetUpdatesHybrid(
      const T& hLeft,
      const T& hRight,
      const T& huLeft,
      const T& huRight,
      const T& bLeft,
      const T& bRight,
      const T& uLeft,
      const T& uRight,
      const T  waveSpeeds[2],
      T&       o_hUpdateLeft,
      T&       o_hUpdateRight,
      T&       o_huUpdateLeft,
      T&       o_huUpdateRight,
      T&       o_maxWaveSpeed
    ) {
      // Store parameters to member variables
      storeParameters(hLeft, hRight, huLeft, huRight, bLeft, bRight, uLeft, uRight);

      computeNetUpdatesWithWaveSpeeds(
        waveSpeeds, o_hUpdateLeft, o_hUpdateRight, o_huUpdateLeft, o_huUpdateRight, o_maxWaveSpeed
      );
    }
  };

} // namespace Solvers
