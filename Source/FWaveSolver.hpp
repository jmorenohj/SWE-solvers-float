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
  class FWaveSolver: public WavePropagationSolver<double> {
  private:
    // Use nondependent names (template base class)
    using WavePropagationSolver<double>::dryTol_;
    using WavePropagationSolver<double>::gravity_;
    using WavePropagationSolver<double>::zeroTol_;

    using WavePropagationSolver<double>::hLeft_;
    using WavePropagationSolver<double>::hRight_;
    using WavePropagationSolver<double>::huLeft_;
    using WavePropagationSolver<double>::huRight_;
    using WavePropagationSolver<double>::bLeft_;
    using WavePropagationSolver<double>::bRight_;
    using WavePropagationSolver<double>::uLeft_;
    using WavePropagationSolver<double>::uRight_;

    using WavePropagationSolver<double>::wetDryState_;

    /**
     * Compute the edge local eigenvalues.
     *
     * @param o_waveSpeeds will be set to: speeds of the linearized waves (eigenvalues).
     */
    void computeWaveSpeeds(__m256d o_waveSpeeds[2]) const {
      // Compute eigenvalues of the Jacobian matrices in states Q_{i-1} and Q_{i}
      __m256d characteristicSpeeds[2]{};

      characteristicSpeeds[0] = _mm256_sub_pd(uLeft_, _mm256_sqrt_pd(_mm256_mul_pd(_mm256_set1_pd(gravity_), hLeft_)));
      characteristicSpeeds[1] = _mm256_add_pd(uRight_, _mm256_sqrt_pd(_mm256_mul_pd(_mm256_set1_pd(gravity_), hRight_)));

      // Compute "Roe speeds"
      
      __m256d hRoe = _mm256_mul_pd(_mm256_set1_pd(0.5), _mm256_add_pd(hRight_, hLeft_));

      __m256d sqrtHL = _mm256_sqrt_pd(hLeft_);
      __m256d sqrtHR = _mm256_sqrt_pd(hRight_);

      __m256d uRoe = _mm256_add_pd(_mm256_mul_pd(uLeft_, sqrtHL), _mm256_mul_pd(uRight_, sqrtHR));

      uRoe = _mm256_div_pd(uRoe, _mm256_add_pd(sqrtHL, sqrtHR));

      __m256d roeSpeeds[2]{};

      roeSpeeds[0] = _mm256_sub_pd(uRoe, _mm256_sqrt_pd(_mm256_mul_pd(_mm256_set1_pd(gravity_), hRoe)));
      roeSpeeds[1] = _mm256_add_pd(uRoe, _mm256_sqrt_pd(_mm256_mul_pd(_mm256_set1_pd(gravity_), hRoe)));

      // Compute eindfeldt speeds
      __m256d einfeldtSpeeds[2]{};
      einfeldtSpeeds[0] = _mm256_min_pd(characteristicSpeeds[0], roeSpeeds[0]);
      einfeldtSpeeds[1] = _mm256_max_pd(characteristicSpeeds[1], roeSpeeds[1]);

      // Set wave speeds
      o_waveSpeeds[0] = einfeldtSpeeds[0];
      o_waveSpeeds[1] = einfeldtSpeeds[1];
    }

    /**
     * Compute the decomposition into f-Waves.
     *
     * @param waveSpeeds speeds of the linearized waves (eigenvalues).
     * @param o_fWaves will be set to: Decomposition into f-Waves.
     */
    void computeWaveDecomposition(const __m256d waveSpeeds[2], __m256d o_fWaves[2][2]) const {
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

      __m256d lambdaDif = _mm256_sub_pd(waveSpeeds[1], waveSpeeds[0]);

      // assert: no division by zero
      //assert(std::abs(lambdaDif) > zeroTol_);

      // Compute the inverse matrix R^{-1}
      __m256d Rinv[2][2]{};

      __m256d oneDivLambdaDif = _mm256_div_pd(_mm256_set1_pd(1.0), lambdaDif);
      Rinv[0][0]        = _mm256_mul_pd(oneDivLambdaDif, waveSpeeds[1]);
      Rinv[0][1]        = _mm256_mul_pd(_mm256_set1_pd(-1.0),oneDivLambdaDif);

      Rinv[1][0] = _mm256_mul_pd(oneDivLambdaDif, _mm256_mul_pd(_mm256_set1_pd(-1.0),waveSpeeds[0]));
      Rinv[1][1] = oneDivLambdaDif;

      // Right hand side
      __m256d fDif[2]{};

      // Calculate modified (bathymetry!) flux difference_mm256_set1_pd(0.5)
      // f(Q_i) - f(Q_{i-1})
      fDif[0] = _mm256_sub_pd(huRight_, huLeft_);
      fDif[1] = _mm256_sub_pd(_mm256_add_pd(_mm256_mul_pd(huRight_,uRight_), _mm256_mul_pd(_mm256_mul_pd(_mm256_mul_pd(_mm256_set1_pd(0.5), _mm256_set1_pd(gravity_)), hRight_), hRight_))
                , _mm256_add_pd(_mm256_mul_pd(huLeft_, uLeft_), _mm256_mul_pd(_mm256_mul_pd(_mm256_mul_pd(_mm256_set1_pd(0.5), _mm256_set1_pd(gravity_)), hLeft_), hLeft_)));

      // \delta x \Psi[2]
      __m256d psi = _mm256_mul_pd(_mm256_mul_pd(_mm256_mul_pd(_mm256_set1_pd(-gravity_), _mm256_set1_pd(0.5)), _mm256_add_pd(hRight_, hLeft_)), _mm256_sub_pd(bRight_, bLeft_));
      fDif[1] = _mm256_sub_pd(fDif[1],psi);

      // Solve linear equations
      __m256d beta[2]{};
      beta[0] = _mm256_add_pd(_mm256_mul_pd(Rinv[0][0], fDif[0]), _mm256_mul_pd(Rinv[0][1], fDif[1]));
      beta[1] = _mm256_add_pd(_mm256_mul_pd(Rinv[1][0], fDif[0]), _mm256_mul_pd(Rinv[1][1], fDif[1]));

      // Return f-waves
      o_fWaves[0][0] = beta[0];
      o_fWaves[0][1] = _mm256_mul_pd(beta[0], waveSpeeds[0]);

      o_fWaves[1][0] = beta[1];
      o_fWaves[1][1] = _mm256_mul_pd(beta[1], waveSpeeds[1]);
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
      const __m256d waveSpeeds[2],
      __m256d&      o_hUpdateLeft,
      __m256d&      o_hUpdateRight,
      __m256d&      o_huUpdateLeft,
      __m256d&      o_huUpdateRight,
      __m256d&      o_maxWaveSpeed
    ) {
      // Reset net updates
      o_hUpdateLeft = o_hUpdateRight = o_huUpdateLeft = o_huUpdateRight = _mm256_setzero_pd();

      //! Where to store the two f-waves
      __m256d fWaves[2][2];

      // Compute the decomposition into f-waves
      computeWaveDecomposition(waveSpeeds, fWaves);

      // Compute the net-updates
      // 1st wave family

      alignas(32) double waveSpeeds04Arr[4];_mm256_storeu_pd(waveSpeeds04Arr, waveSpeeds[0]);
      alignas(32) double waveSpeeds14Arr[4];_mm256_storeu_pd(waveSpeeds14Arr, waveSpeeds[1]);
      alignas(32) double fWaves004Arr[4];_mm256_storeu_pd(fWaves004Arr, fWaves[0][0]);
      alignas(32) double fWaves014Arr[4];_mm256_storeu_pd(fWaves014Arr, fWaves[0][1]);
      alignas(32) double fWaves104Arr[4];_mm256_storeu_pd(fWaves104Arr, fWaves[1][0]);
      alignas(32) double fWaves114Arr[4];_mm256_storeu_pd(fWaves114Arr, fWaves[1][1]);
      alignas(32) double o_hUpdateLeft4Arr[4];_mm256_storeu_pd(o_hUpdateLeft4Arr, o_hUpdateLeft);
      alignas(32) double o_huUpdateLeft4Arr[4];_mm256_storeu_pd(o_huUpdateLeft4Arr, o_huUpdateLeft);
      alignas(32) double o_hUpdateRight4Arr[4];_mm256_storeu_pd(o_hUpdateRight4Arr, o_hUpdateRight);
      alignas(32) double o_huUpdateRight4Arr[4];_mm256_storeu_pd(o_huUpdateRight4Arr, o_huUpdateRight);
      for(int i=0;i<4;i++){
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
      o_hUpdateLeft = _mm256_loadu_pd(o_hUpdateLeft4Arr);
      o_huUpdateLeft = _mm256_loadu_pd(o_huUpdateLeft4Arr);
      o_hUpdateRight = _mm256_loadu_pd(o_hUpdateRight4Arr);
      o_huUpdateRight = _mm256_loadu_pd(o_huUpdateRight4Arr);
      // Compute maximum wave speed (-> CFL-condition)
      o_maxWaveSpeed = _mm256_max_pd(_mm256_andnot_pd(_mm256_set1_pd(-0.0), waveSpeeds[0]), _mm256_andnot_pd(_mm256_set1_pd(-0.0), waveSpeeds[1]));
    }

  protected:
    /**
     * Determine the wet/dry state and set member variables accordingly.
     */
    void determineWetDryState() override {
      // Determine the wet/dry state
      alignas(32) double hLeft4Arr[4];_mm256_storeu_pd(hLeft4Arr, hLeft_);
      alignas(32) double hRight4Arr[4];_mm256_storeu_pd(hRight4Arr, hRight_);
      alignas(32) double uRight4Arr[4];_mm256_storeu_pd(uRight4Arr, uRight_);
      alignas(32) double huRight4Arr[4];_mm256_storeu_pd(huRight4Arr, huRight_);
      alignas(32) double bLeft4Arr[4];_mm256_storeu_pd(bLeft4Arr, bLeft_);
      alignas(32) double bRight4Arr[4];_mm256_storeu_pd(bRight4Arr, bRight_);
      alignas(32) double huLeft4Arr[4];_mm256_storeu_pd(huLeft4Arr, huLeft_);
      alignas(32) double uLeft4Arr[4];_mm256_storeu_pd(uLeft4Arr, uLeft_);
      for(int i=0;i<4;i++){
        if (hLeft4Arr[i] < dryTol_ && hRight4Arr[i] < dryTol_) { // Both cells are dry
          wetDryState_[i] = WavePropagationSolver<double>::WetDryState::DryDry;
        } else if (hLeft4Arr[i] < dryTol_) { // Left cell dry, right cell wet
          uRight4Arr[i] = huRight4Arr[i] / hRight4Arr[i];

          // Set wall boundary conditions.
          // This is not correct in the case of inundation problems.
          hLeft4Arr[i]       = hRight4Arr[i];
          bLeft4Arr[i]       = bRight4Arr[i];
          huLeft4Arr[i]      = -huRight4Arr[i];
          uLeft4Arr[i]       = -uRight4Arr[i];
          wetDryState_[i] = WavePropagationSolver<double>::WetDryState::DryWetWall;
        } else if (hRight4Arr[i] < dryTol_) { // Left cell wet, right cell dry
          uLeft4Arr[i] = huLeft4Arr[i] / hLeft4Arr[i];

          // Set wall boundary conditions.
          // This is not correct in the case of inundation problems.
          hRight4Arr[i]      = hLeft4Arr[i];
          bRight4Arr[i]      = bLeft4Arr[i] ;
          huRight4Arr[i]     = -huLeft4Arr[i];
          uLeft4Arr[i]       = -uRight4Arr[i];
          wetDryState_[i] = WavePropagationSolver<double>::WetDryState::WetDryWall;
        } else { // Both cells wet
          uLeft4Arr[i]  = huLeft4Arr[i] / hLeft4Arr[i];
          uRight4Arr[i] = huRight4Arr[i] / hRight4Arr[i];

          wetDryState_[i] = WavePropagationSolver<double>::WetDryState::WetWet;
        }
      }
      hLeft_ = _mm256_loadu_pd(hLeft4Arr);
      hRight_ = _mm256_loadu_pd(hRight4Arr);
      uRight_ = _mm256_loadu_pd(uRight4Arr);
      huRight_ = _mm256_loadu_pd(huRight4Arr);
      bLeft_ = _mm256_loadu_pd(bLeft4Arr);
      bRight_ = _mm256_loadu_pd(bRight4Arr);
      huLeft_ = _mm256_loadu_pd(huLeft4Arr);
      uLeft_ = _mm256_loadu_pd(uLeft4Arr);
      
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
      double dryTolerance  = static_cast<double>(0.01),
      double gravity       = static_cast<double>(9.81),
      double zeroTolerance = static_cast<double>(0.000000001)
    ):
      WavePropagationSolver<double>(dryTolerance, gravity, zeroTolerance) {}

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
      const __m256d& hLeft,
      const __m256d& hRight,
      const __m256d& huLeft,
      const __m256d& huRight,
      const __m256d& bLeft,
      const __m256d& bRight,
      __m256d&       o_hUpdateLeft,
      __m256d&       o_hUpdateRight,
      __m256d&       o_huUpdateLeft,
      __m256d&       o_huUpdateRight,
      __m256d&       o_maxWaveSpeed
    ) override {
      // Set speeds to zero (will be determined later)
     
      uLeft_ = uRight_ = _mm256_setzero_pd();

      // Reset the maximum wave speed
      o_maxWaveSpeed = _mm256_setzero_pd();

      //! Wave speeds of the f-waves
      /*__m256d waveSpeeds[2];

      // Store parameters to member variables
      WavePropagationSolver<double>::storeParameters(hLeft, hRight, huLeft, huRight, bLeft, bRight);

      // Determine the wet/dry state and compute local variables correspondingly
      determineWetDryState();

      // Compute the wave speeds
      computeWaveSpeeds(waveSpeeds);

      // Use the wave speeds to compute the net-updates
      computeNetUpdatesWithWaveSpeeds(
        waveSpeeds, o_hUpdateLeft, o_hUpdateRight, o_huUpdateLeft, o_huUpdateRight, o_maxWaveSpeed
      );

      // Zero ghost updates (wall boundary)
      alignas(32) double o_hUpdateLeft4Arr[4];_mm256_storeu_pd(o_hUpdateLeft4Arr, o_hUpdateLeft);
      alignas(32) double o_huUpdateLeft4Arr[4];_mm256_storeu_pd(o_huUpdateLeft4Arr, o_huUpdateLeft);
      alignas(32) double o_hUpdateRight4Arr[4];_mm256_storeu_pd(o_hUpdateRight4Arr, o_hUpdateRight);
      alignas(32) double o_huUpdateRight4Arr[4];_mm256_storeu_pd(o_huUpdateRight4Arr, o_huUpdateRight);
      
      for(int i=0;i<4;i++){
        if (wetDryState_[i] == WavePropagationSolver<double>::WetDryState::WetDryWall) {
          o_hUpdateRight4Arr[i]  = 0;
          o_huUpdateRight4Arr[i] = 0;
        } else if (wetDryState_[i] == WavePropagationSolver<double>::WetDryState::DryWetWall) {
          o_hUpdateLeft4Arr[i]  = 0;
          o_huUpdateLeft4Arr[i] = 0;
        }

        // Zero updates and return in the case of dry cells
        if (wetDryState_[i] == WavePropagationSolver<double>::WetDryState::DryDry) {
          o_hUpdateLeft4Arr[i] = o_hUpdateRight4Arr[i] = o_huUpdateLeft4Arr[i] = o_huUpdateRight4Arr[i] = 0.0;
        }
      }*/
      

      
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
