#include "statepredictor.h"

StatePredictor::StatePredictor() {}

MatrixXd StatePredictor::compute_augmented_sigma(
    const VectorXd &current_x,
    const MatrixXd &current_P)
{
  MatrixXd augmented_sigma = MatrixXd::Zero(NAUGMENTED, NSIGMA);
  VectorXd augmented_x = VectorXd::Zero(NAUGMENTED);
  MatrixXd augmented_P = MatrixXd::Zero(NAUGMENTED, NAUGMENTED);

  augmented_x.head(NX) = current_x;

  augmented_P.topLeftCorner(NX, NX) = current_P;
  augmented_P(NX, NX) = VAR_SPEED_NOISE;
  augmented_P(NX + 1, NX + 1) = VAR_YAWRATE_NOISE;

  const MatrixXd L = augmented_P.llt().matrixL();
  augmented_sigma.col(0) = augmented_x;

  for (int c = 0; c < NAUGMENTED; c++)
  {
    const int i = c + 1;
    augmented_sigma.col(i) = augmented_x + SCALE * L.col(c);
    augmented_sigma.col(i + NAUGMENTED) = augmented_x - SCALE * L.col(c);
  }

  return augmented_sigma;
}

VectorXd StatePredictor::fx(const VectorXd &sigma_point, double dt)
{
  const double THRESH = 0.001;

  /*************************************
    * Get the current state
    *************************************/
  const double px = sigma_point(0);
  const double py = sigma_point(1);
  const double speed = sigma_point(2);
  const double yaw = sigma_point(3);
  const double yawrate = sigma_point(4);
  const double speed_noise = sigma_point(5);
  const double yawrate_noise = sigma_point(6);

  /*************************************
    * predict the next state with noise
    * USING THE CTRV MODEL
  *************************************/
  const double cos_yaw = cos(yaw);
  const double sin_yaw = sin(yaw);
  const double dt2 = dt * dt;
  const double p_noise = 0.5 * speed_noise * dt2;   // predicted position noise
  const double y_noise = 0.5 * yawrate_noise * dt2; // predicted yaw noise
  const double dyaw = yawrate * dt;                 // change in yaw
  const double dspeed = speed * dt;                 // change in speed

  const double p_speed = speed + speed_noise * dt;       // predicted speed = assumed constant speed + noise
  const double p_yaw = yaw + dyaw + y_noise;             // predicted yaw
  const double p_yawrate = yawrate + yawrate_noise * dt; // predicted yaw rate = assumed constant yawrate + noise

  double p_px, p_py; // where predicted positions will be stored

  if (fabs(yawrate) <= THRESH)
  {
    // moving straight
    p_px = px + dspeed * cos_yaw + p_noise * cos_yaw;
    p_py = py + dspeed * sin_yaw + p_noise * sin_yaw;
  }
  else
  {
    const double k = speed / yawrate;
    const double theta = yaw + dyaw;
    p_px = px + k * (sin(theta) - sin_yaw) + p_noise * cos_yaw;
    p_py = py + k * (cos_yaw - cos(theta)) + p_noise * sin_yaw;
  }

  VectorXd sigma_x(NX);
  sigma_x << p_px, p_py, p_speed, p_yaw, p_yawrate;

  return sigma_x;
}

MatrixXd StatePredictor::predict_sigma(const MatrixXd &augmented_sigma, double dt)
{
  MatrixXd predicted_sigma(NX, NSIGMA);

  for (int c = 0; c < NSIGMA; ++c)
  {
    predicted_sigma.col(c) = fx(augmented_sigma.col(c), dt);
  }

  return predicted_sigma;
}

VectorXd StatePredictor::predict_x(const MatrixXd &predicted_sigma)
{
  VectorXd predicted_x = VectorXd::Zero(NX);

  for (int c = 0; c < NSIGMA; c++)
  {
    predicted_x += WEIGHTS[c] * predicted_sigma.col(c);
  }

  return predicted_x;
}

MatrixXd StatePredictor::predict_P(const MatrixXd &predicted_sigma, const VectorXd &predicted_x)
{
  MatrixXd predicted_P = MatrixXd::Zero(NX, NX);
  VectorXd dx = VectorXd(NX);

  for (int c = 0; c < NSIGMA; c++)
  {
    dx = predicted_sigma.col(c) - predicted_x;
    dx(3) = normalize(dx(3));
    predicted_P += WEIGHTS[c] * dx * dx.transpose();
  }

  return predicted_P;
}

void StatePredictor::process(VectorXd &current_x, MatrixXd &current_P, double dt)
{
  MatrixXd augmented_sigma = compute_augmented_sigma(current_x, current_P);
  this->sigma = predict_sigma(augmented_sigma, dt);
  this->x = predict_x(this->sigma);
  this->P = predict_P(this->sigma, this->x);
}

MatrixXd StatePredictor::getP() const
{
  return this->P;
}

MatrixXd StatePredictor::get_sigma() const
{
  return this->sigma;
}

VectorXd StatePredictor::getx() const
{
  return this->x;
}
