#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <stdlib.h>
#include "Eigen/Dense"
#include "ground_truth_package.h"
#include "measurement_package.h"
#include "tools.h"

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

void check_arguments(int argc, char* argv[])
{
  string usage_instructions = "Usage instructions: ";
  usage_instructions += argv[0];
  usage_instructions += " path/to/input.txt output.txt";

  bool has_valid_args = false;

  // make sure the user has provided input and output files
  if (argc == 1)
  {
    cerr << usage_instructions << endl;
  }
  else if (argc == 2)
  {
    cerr << "Please include an output file.\n" << usage_instructions << endl;
  }
  else if (argc == 3)
  {
    has_valid_args = true;
  }
  else if (argc > 3)
  {
    cerr << "Too many arguments.\n" << usage_instructions << endl;
  }

  if (!has_valid_args)
  {
    exit(EXIT_FAILURE);
  }
}

void check_files(ifstream& in_file, string& in_name,
                 ofstream& out_file, string& out_name)
{
  if (!in_file.is_open())
  {
    cerr << "Cannot open input file: " << in_name << endl;
    exit(EXIT_FAILURE);
  }

  if (!out_file.is_open())
  {
    cerr << "Cannot open output file: " << out_name << endl;
    exit(EXIT_FAILURE);
  }
}

int main(int argc, char* argv[])
{
  check_arguments(argc, argv);

  string in_file_name_ = argv[1];
  ifstream in_file_(in_file_name_.c_str(), ifstream::in);

  string out_file_name_ = argv[2];
  ofstream out_file_(out_file_name_.c_str(), ofstream::out);

  check_files(in_file_, in_file_name_, out_file_, out_file_name_);

  vector<MeasurementPackage> measurement_pack_list;
  vector<GroundTruthPackage> gt_pack_list;

  string line;

  while (getline(in_file_, line))
  {
    string sensor_type;
    MeasurementPackage meas_package;
    GroundTruthPackage gt_package;
    istringstream iss(line);
    long long timestamp;

    // reads first element from the current line
    iss >> sensor_type;

    if (sensor_type.compare("L") == 0)
    {
      // laser measurement

      // read measurements at this timestamp
      meas_package.sensor_type_ = MeasurementPackage::LASER;
      meas_package.raw_measurements_ = VectorXd(2);
      float px;
      float py;
      iss >> px;
      iss >> py;
      meas_package.raw_measurements_ << px, py;
      iss >> timestamp;
      meas_package.timestamp_ = timestamp;
      measurement_pack_list.push_back(meas_package);
    }
    else if (sensor_type.compare("R") == 0)
    {
      // radar measurement

      // read measurements at this timestamp
      meas_package.sensor_type_ = MeasurementPackage::RADAR;
      meas_package.raw_measurements_ = VectorXd(3);
      float ro;
      float phi;
      float ro_dot;
      iss >> ro;
      iss >> phi;
      iss >> ro_dot;
      meas_package.raw_measurements_ << ro, phi, ro_dot;
      iss >> timestamp;
      meas_package.timestamp_ = timestamp;
      measurement_pack_list.push_back(meas_package);
    }

    // read ground truth data to compare later
    float x_gt;
    float y_gt;
    float vx_gt;
    float vy_gt;
    iss >> x_gt;
    iss >> y_gt;
    iss >> vx_gt;
    iss >> vy_gt;
    gt_package.gt_values_ = VectorXd(4);
    gt_package.gt_values_ << x_gt, y_gt, vx_gt, vy_gt;
    gt_pack_list.push_back(gt_package);
  }

  vector<VectorXd> estimations;
  vector<VectorXd> ground_truth;
  size_t number_of_measurements = measurement_pack_list.size();
  out_file_ << "% px" << "\t";
  out_file_ << "py" << "\t";
  out_file_ << "px_measured" << "\t";
  out_file_ << "py_measured" << "\t";
  out_file_ << "px_true" << "\t";
  out_file_ << "py_true" << "\t";

  int n_x = 5;
  int n_aug = 7;
  double lambda = 3 - n_aug;

  double std_a = 1.5;
  double std_yawdd = 0.55;
  double delta_t;

  //set measurement dimension, radar can measure r, phi, and r_dot
  int n_z_radar = 3;
  double std_radr = 0.3;
  double std_radphi = 0.03;
  double std_radrd = 0.3;

  int n_z_lidar = 2;
  double std_lidar_px = 0.15;
  double std_lidar_py = 0.15;

  //set  state
  VectorXd x = VectorXd(n_x);
  VectorXd x_aug = VectorXd(n_aug);
  VectorXd x_final = VectorXd(n_x);

  // Initialize to available data
  if (measurement_pack_list[0].sensor_type_ == 0)
  {
    VectorXd tmp_z = VectorXd(n_z_lidar);
    tmp_z << measurement_pack_list[0].raw_measurements_;
    x << tmp_z[0], tmp_z[1],0.0, 0.0, 0.0;
  }
  if (measurement_pack_list[0].sensor_type_ == 1)
  {
    VectorXd tmp_z = VectorXd(n_z_radar);
    tmp_z << measurement_pack_list[0].raw_measurements_;
    x << tmp_z[0]*cos(tmp_z[1]), tmp_z[0]*sin(tmp_z[1]),tmp_z[2], 0.0, 0.0;
  }

  //set  covariance matrix
  MatrixXd P = MatrixXd(n_x, n_x);
  MatrixXd P_aug = MatrixXd(n_aug, n_aug);
  MatrixXd P_final = MatrixXd(n_x, n_x);
  P.setIdentity();

  //create sigma point matrix
  MatrixXd Xsig = MatrixXd(n_x, 2 * n_x + 1);
  MatrixXd Xsig_aug = MatrixXd(n_aug, 2 * n_aug + 1);
  MatrixXd Xsig_pred = MatrixXd(n_x, 2 * n_aug + 1);

  long long previous_time = measurement_pack_list[0].timestamp_;
  long long current_time;

  VectorXd ukf_x_cartesian_ = VectorXd(4);

  float x_estimate_ = x(0);
  float y_estimate_ = x(1);
  float vx_estimate_ = x(2) * cos(x(3));
  float vy_estimate_ = x(2) * sin(x(3));
  ukf_x_cartesian_ << x_estimate_, y_estimate_, vx_estimate_, vy_estimate_;
  estimations.push_back(ukf_x_cartesian_);
  ground_truth.push_back(gt_pack_list[0].gt_values_);


//for (size_t k = 0; k < number_of_measurements; ++k) {
  for (size_t k = 1; k < number_of_measurements; ++k)
  {

    bool radar = false;
    bool lidar = false;

    VectorXd z_lidar = VectorXd(n_z_lidar);
    z_lidar.fill(0.0);
    VectorXd z_radar = VectorXd(n_z_radar);
    z_radar.fill(0.0);
    current_time = measurement_pack_list[k].timestamp_;
    delta_t = (current_time-previous_time)/1e6;
    //cout.precision(17);
    //cout << fixed << delta_t << endl;

    if (measurement_pack_list[k].sensor_type_ == 0)
    {
      lidar = true;
      z_lidar << measurement_pack_list[k].raw_measurements_;

    }
    if (measurement_pack_list[k].sensor_type_ == 1)
    {
      radar = true;
      z_radar << measurement_pack_list[k].raw_measurements_;
    }

    //cout << "x" << x << endl;
    //cout << "P" << P << endl;

    //calculate square root of P
    MatrixXd A = P.llt().matrixL();

    //set first column of sigma point matrix
    Xsig.col(0)  = x;

    //set remaining sigma points
    for (int i = 0; i < n_x; i++)
    {
      Xsig.col(i+1)     = x + sqrt(lambda+n_x) * A.col(i);
      Xsig.col(i+1+n_x) = x - sqrt(lambda+n_x) * A.col(i);
    }

    //fill augmented x
    x_aug.head(5) = x;
    x_aug(5) = 0;
    x_aug(6) = 0;

    //create augmented covariance matrix
    P_aug.fill(0.0);
    P_aug.topLeftCorner(5,5) = P;
    P_aug(5,5) = std_a*std_a;
    P_aug(6,6) = std_yawdd*std_yawdd;

    //cout << "P_aug" << P_aug << endl;

    //create square root matrix
    MatrixXd L = P_aug.llt().matrixL();
    //create augmented sigma points
    Xsig_aug.col(0)  = x_aug;
    for (int i = 0; i< n_aug; i++)
    {
      Xsig_aug.col(i+1)       = x_aug + sqrt(lambda+n_aug) * L.col(i);
      Xsig_aug.col(i+1+n_aug) = x_aug - sqrt(lambda+n_aug) * L.col(i);
    }

    //*******************************
    //Predicted sigma points
    //*******************************
    for (int i = 0; i< 2*n_aug+1; i++)
    {
      //extract values for better readability
      double p_x = Xsig_aug(0,i);
      double p_y = Xsig_aug(1,i);
      double v = Xsig_aug(2,i);
      double yaw = Xsig_aug(3,i);
      double yawd = Xsig_aug(4,i);
      double nu_a = Xsig_aug(5,i);
      double nu_yawdd = Xsig_aug(6,i);

      //predicted state values
      double px_p, py_p;

      //avoid division by zero
      if (fabs(yawd) > 0.0001)
      {
        px_p = p_x + v/yawd * ( sin (yaw + yawd*delta_t) - sin(yaw));
        py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw+yawd*delta_t) );
      }
      else
      {
        px_p = p_x + v*delta_t*cos(yaw);
        py_p = p_y + v*delta_t*sin(yaw);
      }

      double v_p = v;
      double yaw_p = yaw + yawd*delta_t;
      double yawd_p = yawd;

      //add noise
      px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
      py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
      v_p = v_p + nu_a*delta_t;

      yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
      yawd_p = yawd_p + nu_yawdd*delta_t;

      //write predicted sigma point into right column
      Xsig_pred(0,i) = px_p;
      Xsig_pred(1,i) = py_p;
      Xsig_pred(2,i) = v_p;
      Xsig_pred(3,i) = yaw_p;
      Xsig_pred(4,i) = yawd_p;
    }

    //*******************************
    //Predicted Mean state and Covariance
    //*******************************
    //create vector for weights
    VectorXd weights = VectorXd(2*n_aug+1);

    //create vector for predicted state
    VectorXd x_kp1 = VectorXd(n_x);

    //create covariance matrix for prediction
    MatrixXd P_kp1 = MatrixXd(n_x, n_x);

    // set weights
    double weight_0 = lambda/(lambda+n_aug);
    weights(0) = weight_0;
    for (int i=1; i<2*n_aug+1; i++)    //2n+1 weights
    {
      double weight = 0.5/(n_aug+lambda);
      weights(i) = weight;
    }

    //predicted state mean
    x_kp1.fill(0.0);
    for (int i = 0; i < 2 * n_aug + 1; i++)    //iterate over sigma points
    {
      x_kp1 = x_kp1+ weights(i) * Xsig_pred.col(i);
    }
    //predicted state covariance matrix
    P_kp1.fill(0.0);
    for (int i = 0; i < 2 * n_aug + 1; i++)    //iterate over sigma points
    {

      // state difference
      VectorXd x_diff = Xsig_pred.col(i) - Xsig_pred.col(0);
      //VectorXd x_diff = Xsig_pred.col(i) - x_kp1;
      //angle normalization
      while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
      while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;
      P_kp1 = P_kp1 + weights(i) * x_diff * x_diff.transpose() ;
    }
    //cout << "max P_kp1: " << P_kp1.maxCoeff() << endl;

    if (radar == true)
    {
      //cout << "Processing Radar" << endl;
      //*******************************
      //Radar Update
      //*******************************
      //create matrix for sigma points in measurement space
      MatrixXd Zsig = MatrixXd(n_z_radar, 2 * n_aug + 1);

      //transform sigma points into measurement space
      for (int i = 0; i < 2 * n_aug + 1; i++)    //2n+1 simga points
      {

        // extract values for better readibility
        double p_x = Xsig_pred(0,i);
        double p_y = Xsig_pred(1,i);
        double v  = Xsig_pred(2,i);
        double yaw = Xsig_pred(3,i);

        double v1 = cos(yaw)*v;
        double v2 = sin(yaw)*v;

        // measurement model
        if (fabs(p_x) < 0.0001 && fabs(p_y) < 0.0001)
        {
          Zsig(0,i) = 0.0;                        //r
          Zsig(1,i) = 0.0;                                 //phi
          Zsig(2,i) = 0.0;   //r_dot
        }
        else
        {
          Zsig(0,i) = sqrt(p_x*p_x + p_y*p_y);                        //r
          Zsig(1,i) = atan2(p_y,p_x);                                 //phi
          Zsig(2,i) = (p_x*v1 + p_y*v2 ) / sqrt(p_x*p_x + p_y*p_y);   //r_dot
        }
      }

      //mean predicted measurement
      VectorXd z_pred = VectorXd(n_z_radar);
      z_pred.fill(0.0);
      for (int i=0; i < 2*n_aug+1; i++)
      {
        z_pred = z_pred + weights(i) * Zsig.col(i);
      }

      //measurement covariance matrix S
      MatrixXd S = MatrixXd(n_z_radar,n_z_radar);
      S.fill(0.0);
      for (int i = 0; i < 2 * n_aug + 1; i++)    //2n+1 simga points
      {
        //residual
        //VectorXd z_diff = Zsig.col(i) - z_pred;
        VectorXd z_diff = Zsig.col(i) - Zsig.col(0);

        //angle normalization
        while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
        while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

        S = S + weights(i) * z_diff * z_diff.transpose();
      }

      //add measurement noise covariance matrix
      MatrixXd R = MatrixXd(n_z_radar,n_z_radar);
      R <<    std_radr*std_radr, 0, 0,
      0, std_radphi*std_radphi, 0,
      0, 0,std_radrd*std_radrd;
      S = S + R;

      //create matrix for cross correlation Tc
      MatrixXd Tc = MatrixXd(n_x, n_z_radar);

      //calculate cross correlation matrix
      Tc.fill(0.0);
      for (int i = 0; i < 2 * n_aug + 1; i++)    //2n+1 simga points
      {

        //residual
        //VectorXd z_diff = Zsig.col(i) - z_pred;
        VectorXd z_diff = Zsig.col(i) - Zsig.col(0);
        //angle normalization
        while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
        while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

        // state difference
        //VectorXd x_diff = Xsig_pred.col(i) - x;
        VectorXd x_diff = Xsig_pred.col(i) - Xsig_pred.col(0);
        //angle normalization
        while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
        while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

        Tc = Tc + weights(i) * x_diff * z_diff.transpose();
      }

      //Kalman gain K;
      MatrixXd K = Tc * S.inverse();

      //residual
      VectorXd z_diff = z_radar - z_pred;

      //angle normalization
      while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
      while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

      //update state mean and covariance matrix
      x_final = x_kp1 + K * z_diff;
      P_final = P_kp1 - K*S*K.transpose();
    }

    if (lidar == true)
    {
      //cout << "Processing Lidar" << endl;
      //*******************************
      //Lidar Update
      //*******************************
      //create matrix for sigma points in measurement space
      MatrixXd Zsig = MatrixXd(n_z_lidar, 2 * n_aug + 1);

      //transform sigma points into measurement space
      for (int i = 0; i < 2 * n_aug + 1; i++)    //2n+1 simga points
      {

        // extract values for better readibility
        double p_x = Xsig_pred(0,i);
        double p_y = Xsig_pred(1,i);

        // measurement model
        Zsig(0,i) = p_x;         //px
        Zsig(1,i) = p_y;         //py
      }

      //mean predicted measurement
      VectorXd z_pred = VectorXd(n_z_lidar);
      z_pred.fill(0.0);
      for (int i=0; i < 2*n_aug+1; i++)
      {
        z_pred = z_pred + weights(i) * Zsig.col(i);
      }

      //measurement covariance matrix S
      MatrixXd S = MatrixXd(n_z_lidar,n_z_lidar);
      S.fill(0.0);
      for (int i = 0; i < 2 * n_aug + 1; i++)    //2n+1 sigma points
      {
        //residual
        //VectorXd z_diff = Zsig.col(i) - z_pred;
        VectorXd z_diff = Zsig.col(i) - Zsig.col(0);

        //angle normalization
        while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
        while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

        S = S + weights(i) * z_diff * z_diff.transpose();
      }

      //add measurement noise covariance matrix
      MatrixXd R = MatrixXd(n_z_lidar,n_z_lidar);
      R <<    std_lidar_px*std_lidar_px, 0,
      0, std_lidar_py*std_lidar_py;
      S = S + R;

      //create matrix for cross correlation Tc
      MatrixXd Tc = MatrixXd(n_x, n_z_lidar);

      //calculate cross correlation matrix
      Tc.fill(0.0);
      for (int i = 0; i < 2 * n_aug + 1; i++)    //2n+1 simga points
      {

        //residual
        //VectorXd z_diff = Zsig.col(i) - z_pred;
        VectorXd z_diff = Zsig.col(i) - Zsig.col(0);
        //angle normalization
        while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
        while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

        // state difference
        //VectorXd x_diff = Xsig_pred.col(i) - x;
        VectorXd x_diff = Xsig_pred.col(i) - Xsig_pred.col(0);
        //angle normalization
        while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
        while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

        Tc = Tc + weights(i) * x_diff * z_diff.transpose();
      }

      //Kalman gain K;
      MatrixXd K = Tc * S.inverse();

      //residual
      VectorXd z_diff = z_lidar - z_pred;

      //angle normalization
      while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
      while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

      //update state mean and covariance matrix
      x_final = x_kp1 + K * z_diff;
      P_final = P_kp1 - K*S*K.transpose();
      //cout << "P_kp1" << P_kp1 << endl;
      //cout << "K" << K << endl;
      //cout << "S" << S << endl;
      //cout << "K*S*K.transpose()" << K*S*K.transpose() << endl;
      //cout << "P_final" << P_final << endl;


    }

    VectorXd ukf_x_cartesian_ = VectorXd(4);

    float x_estimate_ = x_final(0);
    float y_estimate_ = x_final(1);
    float vx_estimate_ = x_final(2) * cos(x_final(3));
    float vy_estimate_ = x_final(2) * sin(x_final(3));

    ukf_x_cartesian_ << x_estimate_, y_estimate_, vx_estimate_, vy_estimate_;

    estimations.push_back(ukf_x_cartesian_);
    ground_truth.push_back(gt_pack_list[k].gt_values_);

    out_file_ << x_final(0) << "\t"; // pos1 - est
    out_file_ << x_final(1) << "\t"; // pos2 - est

    if (radar == true)
    {
      out_file_ << z_radar(0)*cos(z_radar(1)) << "\t"; // pos1 - est
      out_file_ << z_radar(0)*sin(z_radar(1)) << "\t"; // pos1 - est
    }

    if (lidar == true)
    {
      out_file_ << z_lidar(0) << "\t"; // pos1 - est
      out_file_ << z_lidar(1) << "\t"; // pos1 - est
    }

    out_file_ << gt_pack_list[k].gt_values_(0) << "\t";
    out_file_ << gt_pack_list[k].gt_values_(1) << "\n";

    // reset all variables
    x = x_final;
    P = P_final;
    //cout << "x_final" << x_final << endl;
    previous_time = current_time;

  }

  // compute the accuracy (RMSE)
  Tools tools;
  cout << "Accuracy - RMSE:" << endl << tools.CalculateRMSE(estimations, ground_truth) << endl;

  // close files
  if (out_file_.is_open())
  {
    out_file_.close();
  }

  if (in_file_.is_open())
  {
    in_file_.close();
  }

  return 0;
}
