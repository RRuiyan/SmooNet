// update a, Bc, update v1,v2,...

#include <RcppEigen.h>
#include <Rcpp.h>
#include <time.h>
// [[Rcpp::depends(RcppEigen)]]

using namespace Rcpp;
using namespace std;
using namespace Eigen;
//using namespace boost::math;




//[[Rcpp::export]]
List c_smoothing_multiple(MatrixXd Y_mat, MatrixXd Psi,  MatrixXd J_s2, VectorXd nus)
{
  MatrixXd Y_mat_rev=Y_mat;
  for(int k=0; k<Y_mat_rev.cols();k++)
  {
    Y_mat_rev.col(k)=Y_mat.col(k).reverse(); 
  }
  int k1=0, k2=0;
  MatrixXd tmp1=Psi, tmp2=Psi;
  MatrixXd tmp3=Y_mat_rev, tmp4=Y_mat_rev;
  for(int k=0; k<Psi.rows();k++)
  {
    if(k%2==1)
    {
      tmp1.row(k1)=Psi.row(k);
      tmp3.row(k1)=Y_mat_rev.row(k);
      k1=k1+1;
    }
    else
    {
      tmp2.row(k2)=Psi.row(k);
      tmp4.row(k2)=Y_mat_rev.row(k);
      k2=k2+1;
    }
  }
  
  
  MatrixXd Psi_1=tmp1.topRows(k1), Psi_2=tmp2.topRows(k2); 
  MatrixXd Y1=tmp3.topRows(k1), Y2=tmp4.topRows(k2); 
  MatrixXd cv_err(nus.size(), Y2.cols());
  for(int k=0; k<nus.size();k++)
  {
    double nu=nus(k);
    MatrixXd A=Psi_1.transpose()*Psi_1+Psi_1.rows()*nu*J_s2;
    MatrixXd W_mat= A.colPivHouseholderQr().solve(Psi_1.transpose()*Y1);
    cv_err.row(k)=(Psi_2*W_mat-Y2).colwise().squaredNorm();
  }
  
  Eigen::Index minRow;
  MatrixXd nonzero_ind=MatrixXd::Zero(cv_err.rows(), cv_err.cols());
  for(int k=0; k<cv_err.cols();k++)
  {
    double tmp_min=cv_err.col(k).minCoeff(&minRow);
    nonzero_ind(minRow, k)=1;
  }
  MatrixXd W=MatrixXd::Zero(Psi.cols(), Y_mat_rev.cols());
  for(int k=0; k<nus.size();k++)
  {
    double nu=nus(k);
    MatrixXd A=Psi.transpose()*Psi+Psi.rows()*nu*J_s2;
    MatrixXd tmp_mat=Y_mat_rev;
    for(int m=0; m<cv_err.cols();m++)
    { 
      tmp_mat.col(m)=tmp_mat.col(m)*nonzero_ind(k, m);
    }
    W=W+A.colPivHouseholderQr().solve(Psi.transpose()*tmp_mat); 
  }
  
  return List::create(_["cv_err"]=cv_err,  _["W"]=W);
}


//[[Rcpp::export]]
List c_smoothing_single(VectorXd Y, MatrixXd Psi,  MatrixXd J_s2, VectorXd nus)
{
  
  VectorXd Y_rev=Y.reverse(); 
  int k1=0, k2=0;
  MatrixXd tmp1=Psi, tmp2=Psi;
  VectorXd tmp3=Y, tmp4=Y;
  for(int k=0; k<Y.size();k++)
  {
    if(k%2==1)
    {
      tmp1.row(k1)=Psi.row(k);
      tmp3(k1)=Y_rev(k);
      k1=k1+1;
    }
    else
    {
      tmp2.row(k2)=Psi.row(k);
      tmp4(k2)=Y_rev(k);
      k2=k2+1;
    }
  }
  
  
  MatrixXd Psi_1=tmp1.topRows(k1), Psi_2=tmp2.topRows(k2); 
  VectorXd Y1=tmp3.head(k1), Y2=tmp4.head(k2); 
  VectorXd cv_err(nus.size()); 
  for(int k=0; k<nus.size();k++)
  {
    double nu=nus(k);
    MatrixXd A=Psi_1.transpose()*Psi_1+Psi_1.rows()*nu*J_s2;
    VectorXd w= A.colPivHouseholderQr().solve(Psi_1.transpose()*Y1);
    cv_err(k)=(Psi_2*w-Y2).squaredNorm();
  }
  Eigen::Index min_ele;
  double tmp_min=cv_err.minCoeff(&min_ele);
  double nu=nus(min_ele);
  MatrixXd A=Psi.transpose()*Psi+Psi.rows()*nu*J_s2;
  VectorXd w= A.colPivHouseholderQr().solve(Psi.transpose()*Y_rev); 
  
  
  return List::create(_["cv_err"]=cv_err, _["w"]=w);
}




//[[Rcpp::export]]
List c_obj_val(VectorXd a, MatrixXd B, MatrixXd V, VectorXd c, MatrixXd seg_end_index_C,
               VectorXd delta_Y, MatrixXd Z, MatrixXd Phi, VectorXd mu_vec, MatrixXd J_s0,
               MatrixXd J_u0, MatrixXd J_s2, MatrixXd J_u2,VectorXd lambdas)
{
  //Phi: Nu*pu, B:pu*J, V: ps*J, Z: nseg*ps
  int J=B.cols();
  int n_seg=seg_end_index_C.rows();
  double lambda_1=lambdas[0], lambda_2=lambdas[1];
  //Rcout << "n_seg= " << n_seg<< std::endl;
  
  MatrixXd tmp_mat1=Phi*B*V.transpose()*Z.transpose(); //Nu*nseg
  VectorXd tmp_vec1=Phi*c;
  MatrixXd sig_input=tmp_mat1.colwise()+tmp_vec1;
  
  
  MatrixXd tmp_sig_1=MatrixXd::Constant(sig_input.rows(), sig_input.cols(), 50);
  MatrixXd tmp_sig_2=MatrixXd::Constant(sig_input.rows(), sig_input.cols(), -50);
  sig_input=sig_input.array().min(tmp_sig_1.array());
  sig_input=sig_input.array().max(tmp_sig_2.array()); 
  sig_input=sig_input.array().exp();
  
  MatrixXd sigmoid_val_all=sig_input.array()/(1+sig_input.array());
  tmp_vec1=sigmoid_val_all.transpose()*(Phi*a); //nseg*1
  // Rcout << "dim of tmp_vec1= " << tmp_vec1.rows()<<" "<< tmp_vec1.cols()<< std::endl;
  VectorXd d=delta_Y;
  for(int j=0; j<n_seg; j++)
  {
    d(j)-=tmp_vec1.segment(seg_end_index_C(j,0), seg_end_index_C(j,1)-seg_end_index_C(j,0)+1).sum()*mu_vec(j);
  }
  
  double L=d.squaredNorm()/n_seg;
  double P=lambda_1*(a.dot(J_u2*a)+c.dot(J_u2*c));
  for(int j=0; j<J; j++)
  {
    P+=lambda_2*V.col(j).dot(J_s2*V.col(j))*B.col(j).dot(J_u0*B.col(j))+
      lambda_2*V.col(j).dot(J_s0*V.col(j))*B.col(j).dot(J_u2*B.col(j));
  }
  double obj_val=L+P;
  return List::create(_["obj_val"]=obj_val, _["d"]=d, _["sigmoid_val_all"]=sigmoid_val_all);
}





//[[Rcpp::export]]
List c_getDrv_a(VectorXd a, MatrixXd B, MatrixXd V, VectorXd c, MatrixXd seg_end_index_C,
                VectorXd delta_Y, MatrixXd Z, MatrixXd Phi, VectorXd mu_vec, MatrixXd J_s0,
                MatrixXd J_u0, MatrixXd J_s2, MatrixXd J_u2,VectorXd lambdas)
{
  
  // int J=B.cols();
  int nbasis_u=Phi.cols();
  int n_seg=seg_end_index_C.rows();
  double lambda_1=lambdas[0]; //, lambda_2=lambdas[1];
  
  List obj_list=c_obj_val(a, B, V, c, seg_end_index_C, delta_Y, Z, Phi, mu_vec, J_s0,
                          J_u0, J_s2,  J_u2,lambdas);
  MatrixXd sigmoid_val_all=obj_list["sigmoid_val_all"];
  VectorXd d=obj_list["d"];
  
  
  VectorXd dLda=VectorXd::Zero(nbasis_u);
  MatrixXd tmp_mat1=Phi.transpose()*sigmoid_val_all; 
  MatrixXd d2Ld2a=MatrixXd::Zero(nbasis_u, nbasis_u);
  // Rcout << "dim of tmp_mat1= " << tmp_mat1.rows()<<" "<< tmp_mat1.cols()<< std::endl;
  for(int j=0; j<n_seg; j++)
  {
    VectorXd tmp_vec1=tmp_mat1.middleCols(seg_end_index_C(j,0), seg_end_index_C(j,1)-seg_end_index_C(j,0)+1).rowwise().sum()*mu_vec(j);
    dLda-= tmp_vec1*d(j);
    d2Ld2a+=tmp_vec1*tmp_vec1.transpose();
  }
  
  dLda=dLda*2/n_seg;
  d2Ld2a=d2Ld2a*2/n_seg;
  
  VectorXd dPLda=dLda+2*lambda_1*(J_u2*a);
  MatrixXd d2PLd2a=d2Ld2a+2*lambda_1*J_u2;
  
  return List::create(_["dPLda"]=dPLda, _["d2PLd2a"]=d2PLd2a, _["obj_list"]=obj_list);
}




// [[Rcpp::export]]
List c_getDrv_B(VectorXd a, MatrixXd B, MatrixXd V, VectorXd c, MatrixXd seg_end_index_C,
                VectorXd delta_Y, MatrixXd Z, MatrixXd Phi, VectorXd mu_vec, MatrixXd J_s0,
                MatrixXd J_u0, MatrixXd J_s2, MatrixXd J_u2,VectorXd lambdas)
{
  
  int J=B.cols();
  int nbasis_u=Phi.cols();
  int Nu=Phi.rows(); 
  int n_seg=seg_end_index_C.rows();
  double lambda_1=lambdas[0], lambda_2=lambdas[1];  
  
  List obj_list=c_obj_val(a, B, V, c, seg_end_index_C, delta_Y, Z, Phi, mu_vec, J_s0,
                          J_u0, J_s2,  J_u2,lambdas);
  MatrixXd sigmoid_val_all=obj_list["sigmoid_val_all"];
  MatrixXd sigmoid_d1_all=sigmoid_val_all.array() * (1-sigmoid_val_all.array());
  MatrixXd sigmoid_d2_all=sigmoid_d1_all.array() * (1-2*sigmoid_val_all.array());
  VectorXd d=obj_list["d"], Phia=Phi*a;
  MatrixXd ZV=Z*V;  
  
  VectorXd dPLdB=VectorXd::Zero(J*nbasis_u);  
  MatrixXd dDLdB=MatrixXd::Zero(Nu, J);
  MatrixXd dDdB=MatrixXd::Zero(Nu, J);
  MatrixXd d2DLd2B_1=MatrixXd::Zero(J*Nu, J*Nu);   
  MatrixXd d2Ld2B_1=MatrixXd::Zero(J*nbasis_u, J*nbasis_u);
  MatrixXd d2Ld2B_2=MatrixXd::Zero(J*nbasis_u, J*nbasis_u);  
  VectorXd dPdB=VectorXd::Zero(J*nbasis_u); 
  MatrixXd d2Pd2B=MatrixXd::Zero(J*nbasis_u, J*nbasis_u); 
  
  MatrixXd wbb=MatrixXd::Zero(Nu, J*J);
  int m, n; 
  MatrixXd tmp_mat3;
  for(int j=0; j<n_seg; j++)
  {
    double mu=mu_vec(j);
    MatrixXd sigmoid_j=sigmoid_val_all.middleCols(seg_end_index_C(j,0), seg_end_index_C(j,1)-seg_end_index_C(j,0)+1);
    MatrixXd sigmoid_d1_j=sigmoid_d1_all.middleCols(seg_end_index_C(j,0), seg_end_index_C(j,1)-seg_end_index_C(j,0)+1);
    MatrixXd sigmoid_d2_j=sigmoid_d2_all.middleCols(seg_end_index_C(j,0), seg_end_index_C(j,1)-seg_end_index_C(j,0)+1);
    VectorXd sigmoid_j_rowsum=sigmoid_j.rowwise().sum();
    VectorXd sigmoid_d1_j_rowsum=sigmoid_d1_j.rowwise().sum();
    VectorXd sigmoid_d2_j_rowsum=sigmoid_d2_j.rowwise().sum();
    MatrixXd tmp_mat1=ZV.middleRows(seg_end_index_C(j,0), seg_end_index_C(j,1)-seg_end_index_C(j,0)+1);
    MatrixXd sigmoid_d1_j_ZV=sigmoid_d1_j*tmp_mat1;
    MatrixXd sigmoid_d2_j_ZV=sigmoid_d2_j*tmp_mat1;
    for(m=0; m<J; m++)//for derivatives for bj
    {
      dDdB.col(m)=-mu*sigmoid_d1_j_ZV.col(m); 
      for(n=0; n<J; n++)
      { 
        wbb.col(m*J+n)=wbb.col(m*J+n)-mu*d(j)*sigmoid_d2_j*(tmp_mat1.col(m).cwiseProduct(tmp_mat1.col(n)));
      }
    }
    
    dDLdB=dDLdB+dDdB*d(j)*2/n_seg;
    Map<VectorXd> tmp_mat2(dDdB.data(), dDdB.size());
    d2DLd2B_1=d2DLd2B_1+tmp_mat2*tmp_mat2.transpose()*2/n_seg; //product of 1st derivatives
  }
  
  VectorXd vj, bj;
  for(m=0; m<J; m++)//for derivatives for bj
  { 
    vj=V.col(m);
    d2Pd2B.block(m*nbasis_u, m*nbasis_u, nbasis_u, nbasis_u)=
      2*lambda_2*vj.dot(J_s2*vj)*J_u0+2*lambda_2*vj.dot(J_s0*vj)*J_u2;
    for(n=0; n<J; n++)
    {  
      tmp_mat3=d2DLd2B_1.block(m*Nu, n*Nu, Nu, Nu);
      d2Ld2B_1.block(m*nbasis_u, n*nbasis_u, nbasis_u, nbasis_u)=Phi.transpose()*(tmp_mat3.cwiseProduct(Phia*Phia.transpose()))*Phi;
      d2Ld2B_2.block(m*nbasis_u, n*nbasis_u, nbasis_u, nbasis_u)=Phi.transpose()*((Phia.cwiseProduct(wbb.col(m*J+n))).asDiagonal()*Phi);
    }
  }
  
  VectorXd tmp_vec2;
  for(m=0; m<J; m++)//for derivatives for bj
  { 
    bj=B.col(m);
    dPLdB.segment(m*nbasis_u, nbasis_u)=Phi.transpose()*(Phia.cwiseProduct(dDLdB.col(m)))+
      d2Pd2B.block(m*nbasis_u, m*nbasis_u, nbasis_u, nbasis_u)*bj;
  }
  
  
  MatrixXd d2PLd2B_part=(d2Ld2B_1+d2Ld2B_1.transpose())/2+(d2Pd2B+d2Pd2B.transpose())/2; 
  MatrixXd d2PLd2B=d2PLd2B_part+ (d2Ld2B_2+d2Ld2B_2.transpose())/n_seg;
  
  
  return List::create(_["dPLdB"]=dPLdB, _["d2PLd2B_part"]=d2PLd2B_part, _["d2PLd2B"]=d2PLd2B,
                      _["obj_list"]=obj_list);
}




// [[Rcpp::export]]
List c_getDrv_vj(int jcmp, VectorXd a, MatrixXd B, MatrixXd V, VectorXd c, MatrixXd seg_end_index_C,
                 VectorXd delta_Y, MatrixXd Z, MatrixXd Phi, VectorXd mu_vec, MatrixXd J_s0,
                 MatrixXd J_u0, MatrixXd J_s2, MatrixXd J_u2,VectorXd lambdas)
{
  
  //  int J=B.cols();
  VectorXd vj=V.col(jcmp);
  VectorXd bj=B.col(jcmp);
  //int nbasis_u=Phi.cols();
  int nbasis_s=V.rows();
  int n_seg=seg_end_index_C.rows();
  //  double lambda_1=lambdas[0], 
  double lambda_2=lambdas[1];
  
  List obj_list=c_obj_val(a, B, V, c, seg_end_index_C, delta_Y, Z, Phi, mu_vec, J_s0,
                          J_u0, J_s2,  J_u2,lambdas);
  MatrixXd sigmoid_val_all=obj_list["sigmoid_val_all"];
  MatrixXd sigmoid_d1_all=sigmoid_val_all.array() * (1-sigmoid_val_all.array());
  MatrixXd sigmoid_d2_all=sigmoid_d1_all.array() * (1-2*sigmoid_val_all.array());
  VectorXd d=obj_list["d"];
  
  
  VectorXd dLdvj=VectorXd::Zero(nbasis_s);
  MatrixXd d2Ld2vj=MatrixXd::Zero(nbasis_s, nbasis_s);
  MatrixXd d2Ld2vj_1=MatrixXd::Zero(nbasis_s, nbasis_s);
  MatrixXd d2Ld2vj_2=MatrixXd::Zero(nbasis_s, nbasis_s); 
  VectorXd tmp_vec2=sigmoid_d1_all.transpose()*((Phi*a).cwiseProduct(Phi*bj)); 
  VectorXd tmp_vec4=sigmoid_d2_all.transpose()*((Phi*a).cwiseProduct(Phi*bj).cwiseProduct(Phi*bj)); 
  
  // Rcout << "dim of tmp_mat1= " << tmp_mat1.rows()<<" "<< tmp_mat1.cols()<< std::endl;
  for(int j=0; j<n_seg; j++)
  {
    MatrixXd tmp_vec5=Z.middleRows(seg_end_index_C(j,0), seg_end_index_C(j,1)-seg_end_index_C(j,0)+1);
    VectorXd tmp_vec3=tmp_vec5.transpose()*tmp_vec2.segment(seg_end_index_C(j,0), seg_end_index_C(j,1)-seg_end_index_C(j,0)+1)*mu_vec(j);
    dLdvj-=tmp_vec3*d(j);
    d2Ld2vj_1+=tmp_vec3*tmp_vec3.transpose();
    d2Ld2vj_2-=tmp_vec5.transpose()*(tmp_vec4.segment(seg_end_index_C(j,0), seg_end_index_C(j,1)-seg_end_index_C(j,0)+1).asDiagonal()*tmp_vec5)*mu_vec(j)*d(j);
  }
  
  MatrixXd d2Pd2vj=2*lambda_2*bj.dot(J_u2*bj)*J_s0+
    2*lambda_2*bj.dot(J_u0*bj)*J_s2;
  dLdvj=dLdvj*2/n_seg; 
  VectorXd dPLdvj=dLdvj+d2Pd2vj*vj; 
  MatrixXd tmp_mat=d2Ld2vj_1*2/n_seg+d2Pd2vj;
  MatrixXd d2PLd2vj_part=(tmp_mat+tmp_mat.transpose())/2;
  tmp_mat=d2PLd2vj_part+d2Ld2vj_2*2/n_seg;
  MatrixXd d2PLd2vj=(tmp_mat+tmp_mat.transpose())/2;
  
  return List::create(_["dPLdvj"]=dPLdvj,   _["d2PLd2vj_part"]=d2PLd2vj_part, _["d2PLd2vj"]=d2PLd2vj,
                      _["obj_list"]=obj_list);
}



// [[Rcpp::export]]
List c_getDrv_c(VectorXd a, MatrixXd B, MatrixXd V, VectorXd c, MatrixXd seg_end_index_C,
                VectorXd delta_Y, MatrixXd Z, MatrixXd Phi, VectorXd mu_vec, MatrixXd J_s0,
                MatrixXd J_u0, MatrixXd J_s2, MatrixXd J_u2,VectorXd lambdas)
{
  
  int nbasis_u=Phi.cols();
  int n_seg=seg_end_index_C.rows();
  double lambda_1=lambdas[0]; 
  
  List obj_list=c_obj_val(a, B, V, c, seg_end_index_C, delta_Y, Z, Phi, mu_vec, J_s0,
                          J_u0, J_s2,  J_u2,lambdas);
  MatrixXd sigmoid_val_all=obj_list["sigmoid_val_all"];
  MatrixXd sigmoid_d1_all=sigmoid_val_all.array() * (1-sigmoid_val_all.array());
  MatrixXd sigmoid_d2_all=sigmoid_d1_all.array() * (1-2*sigmoid_val_all.array());
  VectorXd d=obj_list["d"];
  
  
  VectorXd dLdc=VectorXd::Zero(nbasis_u);
  MatrixXd d2Ld2c=MatrixXd::Zero(nbasis_u, nbasis_u);
  MatrixXd d2Ld2c_1=MatrixXd::Zero(sigmoid_d1_all.rows(), sigmoid_d1_all.rows());
  VectorXd g=VectorXd::Zero(sigmoid_d1_all.rows()); 
  VectorXd tmp_vec3=VectorXd::Zero(sigmoid_d1_all.rows());
  
  for(int j=0; j<n_seg; j++)
  {
    VectorXd tmp_vec4=sigmoid_d1_all.middleCols(seg_end_index_C(j,0), seg_end_index_C(j,1)-seg_end_index_C(j,0)+1).rowwise().sum()*mu_vec(j);
    //Rcout << "dim of tmp_vec4= " << tmp_vec4.rows()<<" "<< tmp_vec4.cols()<< std::endl;
    tmp_vec3-=tmp_vec4*d(j);  
    d2Ld2c_1+=tmp_vec4*tmp_vec4.transpose();
    g-=sigmoid_d2_all.middleCols(seg_end_index_C(j,0), seg_end_index_C(j,1)-seg_end_index_C(j,0)+1).rowwise().sum()*mu_vec(j)*d(j);  
  }
  VectorXd tmp_vec2=Phi*a;
  dLdc=Phi.transpose()*(tmp_vec2.cwiseProduct(tmp_vec3))*2/n_seg;
  
  MatrixXd d2Ld2c_part=Phi.transpose()*(d2Ld2c_1.cwiseProduct(tmp_vec2*tmp_vec2.transpose()))*Phi*2/n_seg;
  d2Ld2c=d2Ld2c_part+Phi.transpose()*((tmp_vec2.cwiseProduct(g)).asDiagonal()*Phi)*2/n_seg;
  
  VectorXd dPLdc=dLdc+2*lambda_1*(J_u2*c);
  MatrixXd tmp_mat=d2Ld2c_part+2*lambda_1*J_u2;
  MatrixXd d2PLd2c_part=(tmp_mat+tmp_mat.transpose())/2;
  tmp_mat=d2Ld2c+2*lambda_1*J_u2;
  MatrixXd d2PLd2c=(tmp_mat+tmp_mat.transpose())/2;
  
  return List::create(_["dPLdc"]=dPLdc, _["d2PLd2c_part"]=d2PLd2c_part, _["d2PLd2c"]=d2PLd2c,
                      _["obj_list"]=obj_list);
}



//[[Rcpp::export]]
List update_B(VectorXd a, MatrixXd B0, MatrixXd V, VectorXd c, MatrixXd seg_end_index_C,
              VectorXd delta_Y, MatrixXd Z, MatrixXd Phi, VectorXd mu_vec, MatrixXd J_s0,
              MatrixXd J_u0, MatrixXd J_s2, MatrixXd J_u2,VectorXd lambdas, int ini_method_ind)
{
  int J=B0.cols();
  int nbasis_u=Phi.cols();
  int nbasis_s=V.rows();
  int Nu=Phi.rows(); 
  int n_seg=seg_end_index_C.rows();
  
  VectorXd dB_vec=VectorXd::Zero(J*nbasis_u);;
  MatrixXd B=B0, B_new;
  Map<VectorXd> B_vec(B.data(), J*nbasis_u);
  
  List obj_list, obj_list_new;
  double obj_val;  
  int change_method_ind=ini_method_ind; 
  double delta=0;
  for(int count=0; count<10; count++)
  { 
    //Rcout << "-------------------------"<< std::endl;
    //Rcout << "in update_B count=" << count << std::endl;
    //    Rcout << "a.head=" << a.head(5) << " c.head=" << c.head(5) << std::endl;
    //    Rcout << "B=" << B.block(0,0,5,5) << std::endl;
    //    Rcout << "V=" << V.block(0,0,5,5) << std::endl;
    List drv_B_ret=c_getDrv_B(a, B, V, c, seg_end_index_C,
                              delta_Y, Z, Phi, mu_vec, J_s0,
                              J_u0, J_s2, J_u2, lambdas);
    
    obj_list=drv_B_ret["obj_list"];
    obj_val=obj_list["obj_val"];
    // Rcout << "inside update, obj_val before Bac="<< obj_val  << std::endl; 
    VectorXd dPLdB=drv_B_ret["dPLdB"];
    MatrixXd d2PLd2B=drv_B_ret["d2PLd2B"];
    MatrixXd d2PLd2B_part=drv_B_ret["d2PLd2B_part"];
    MatrixXd id_mtx=MatrixXd::Identity(d2PLd2B_part.rows(), d2PLd2B_part.cols());
    
    //Rcout << "obj_val="<< obj_val  << std::endl; 
    //Rcout << "delta="<< delta << std::endl; 
    //Rcout << "the norm of dPLdB="<< dPLdB.norm() << std::endl; 
    
    
    
    if(change_method_ind==0)
    {
      dB_vec=-(d2PLd2B_part+1e-5*id_mtx).ldlt().solve(dPLdB);
      int dd=dB_vec.array().isNaN().any()*1;
      if(dd==1)
      {
        dB_vec=-d2PLd2B.ldlt().solve(dPLdB);
        change_method_ind=1;
      }
    }
    else
    {
      dB_vec=-(d2PLd2B+delta*id_mtx).colPivHouseholderQr().solve(dPLdB);
    }
    Map<MatrixXd> dB_mtx(dB_vec.data(), nbasis_u, J);
    
    B_new=B+dB_mtx;
    obj_list_new=c_obj_val(a, B_new, V, c, seg_end_index_C, delta_Y, Z, Phi, mu_vec, J_s0,
                           J_u0, J_s2,  J_u2, lambdas);
    double obj_val_new=obj_list_new["obj_val"];
    //Rcout << "obj_val_new-obj_val="<< obj_val_new-obj_val << std::endl; 
    
    if(isnan(obj_val_new)==1){
      obj_val_new=1e30;
    }
    
    int back_fail=0, been_back=0;
    
    if(obj_val<obj_val_new)
    {
      
      for(int k=0; k<5; k++)
      {
        dB_mtx=dB_mtx/3; //3 changed to 4
        B_new=B+dB_mtx;
        List obj_list_tmp=c_obj_val(a, B_new, V, c, seg_end_index_C, delta_Y, Z, Phi, mu_vec, J_s0,
                                    J_u0, J_s2,  J_u2, lambdas);
        double obj_val_tmp=obj_list_tmp["obj_val"];
        //Rcout << "back k=" << k << ", obj_val_tmp=" << obj_val_tmp << std::endl;
        been_back=k;
        if(obj_val>obj_val_tmp){
          back_fail=0;
          obj_val_new=obj_val_tmp;
          obj_list_new=obj_list_tmp;
          //       Rcout << "back k=" << k << " new_value=" << obj_val_new << std::endl;
          break;
        }
        back_fail=1;
        
      }
    }
    
    if((back_fail==0) & (obj_val-obj_val_new>1e-4*obj_val))
    {
      
      if(((obj_val-obj_val_new<0.01*obj_val) | (been_back>0))&(change_method_ind==0))
      {
        //Rcout << "been_back=" << been_back << std::endl;
        change_method_ind=1;
        EigenSolver<MatrixXd> es(d2PLd2B, false);
        VectorXd eigen_values=es.eigenvalues().real();
        //Rcout << "Iniyially, the eigenvalues of d2PLd2B :" << eigen_values.minCoeff()<<", "<< eigen_values.maxCoeff() << std::endl; 
        double eig_min=eigen_values.minCoeff(), eig_max=eigen_values.maxCoeff();
        if(eig_min<0)
        {
          delta=-2*eig_min;
          if(delta<eig_max*1e-5)
          {
            delta=eig_max*1e-5;
          }
        }
        else
        {
          if(eig_min<eig_max*1e-5)
          {
            delta=eig_max*1e-5;
          }
        }
      }
      
      //Rcout << "obj_val_new="<< obj_val_new  << std::endl; 
      obj_val=obj_val_new;
      obj_list=obj_list_new;
      B=B_new;
    }
    else
    {
      if(obj_val>obj_val_new)
      {
        obj_val=obj_val_new;
        obj_list=obj_list_new;
        B=B_new;
      } 
      break; 
    }
  }//end for/
  return List::create(_["B"]=B, _["obj_list"]=obj_list);
}






//[[Rcpp::export]]
List update_vj(int jcmp, VectorXd a, MatrixXd B, MatrixXd V0, VectorXd c, MatrixXd seg_end_index_C,
               VectorXd delta_Y, MatrixXd Z, MatrixXd Phi, VectorXd mu_vec, MatrixXd J_s0,
               MatrixXd J_u0, MatrixXd J_s2, MatrixXd J_u2,VectorXd lambdas, MatrixXd Proj)
{
  // Given XPhi, Y, basis, ncomp=1 and tuning parameters, find minimizer of penalized SSE
  // int J=B.cols();
  VectorXd vj=V0.col(jcmp);
  VectorXd bj=B.col(jcmp);
  // int nbasis_u=Phi.cols();
  // int n_seg=seg_end_index_C.rows();
  // double lambda_1=lambdas[0], lambda_2=lambdas[1];
  
  int back_fail=0;
  
  MatrixXd V=V0; 
  List obj_list;
  double obj_val;  
  List  obj_list_new;
  // int change_method_ind=0; 
  //Rcout << "-------------------------"<< std::endl;
  //Rcout << "jcmp=" << jcmp << std::endl;
  for(int count=0; count<20; count++)
  { 
    //  Rcout << "count=" << count << std::endl;
    List drvV_ret=c_getDrv_vj(jcmp, a, B, V, c, seg_end_index_C,
                              delta_Y, Z, Phi, mu_vec, J_s0,
                              J_u0, J_s2, J_u2, lambdas);
    
    VectorXd dPLdvj=drvV_ret["dPLdvj"];
    MatrixXd d2PLd2vj=drvV_ret["d2PLd2vj"]; 
    obj_list=drvV_ret["obj_list"];
    obj_val=obj_list["obj_val"];
    
    
    //Rcout <<  "count=" << count << "obj_val=" << obj_val << std::endl;
    VectorXd dvj;
    MatrixXd V_new=V;
    VectorXd d1=Proj.transpose()*dPLdvj;
    MatrixXd D2=(Proj.transpose()*d2PLd2vj)*Proj;
    dvj=-Proj*(D2.householderQr().solve(d1));  
    
    EigenSolver<MatrixXd> es(D2, false);
    VectorXd eigen_values=es.eigenvalues().real();
    //Rcout << "The eigenvalues of D2 :" << eigen_values.minCoeff()<<", "<< eigen_values.maxCoeff() << std::endl; 
    
    //Rcout << "dvj.norm=" << dvj.norm() << " V.col.norm=" << V.col(jcmp).norm() << std::endl;
    
    V_new.col(jcmp)=V.col(jcmp)+dvj;
    
    obj_list_new=c_obj_val(a, B, V_new, c, seg_end_index_C, delta_Y, Z, Phi, mu_vec, J_s0,
                           J_u0, J_s2, J_u2, lambdas);
    
    double obj_val_new=obj_list_new["obj_val"];
    
    
    //Rcout << "obj_val_new=" << obj_val_new << std::endl;
    
    if(isnan(obj_val_new)==1)
    {
      obj_val_new=1e30;
    }
    
    back_fail=0;
    if(obj_val<obj_val_new)
    { 
      
      for(int k=0; k<4; k++) // 4 changed to 8
      {
        dvj=dvj/3;  //3 changed to 4
        V_new.col(jcmp)=V.col(jcmp)+dvj;
        List obj_list_tmp=c_obj_val(a, B, V_new, c, seg_end_index_C, delta_Y, Z, Phi, mu_vec, J_s0,
                                    J_u0, J_s2, J_u2, lambdas);
        double obj_val_tmp=obj_list_tmp["obj_val"];
        //Rcout << "back k=" << k << " obj_val_tmp=" << obj_val_tmp << std::endl;
        if(obj_val>obj_val_tmp)
        {
          back_fail=0;
          obj_val_new=obj_val_tmp;
          obj_list_new=obj_list_tmp; 
          break;
        }
        back_fail=1;
        
      }
    }
    
    
    if((back_fail==0)&(obj_val-obj_val_new>1e-5*obj_val))
    { 
      obj_val=obj_val_new;
      obj_list=obj_list_new;
      V=V_new; 
    }
    else
    {
      if(obj_val>obj_val_new)
      {
        obj_val=obj_val_new;
        obj_list=obj_list_new;
        V=V_new; 
      }
      break;
    }
  }
  
  
  return List::create(_["V"]=V, _["obj_list"]=obj_list);
}




//[[Rcpp::export]]
List update_c(VectorXd a, MatrixXd B, MatrixXd V, VectorXd c0, MatrixXd seg_end_index_C,
              VectorXd delta_Y, MatrixXd Z, MatrixXd Phi, VectorXd mu_vec, MatrixXd J_s0,
              MatrixXd J_u0, MatrixXd J_s2, MatrixXd J_u2,VectorXd lambdas, int ini_method_ind)
{
  // Given XPhi, Y, basis, ncomp=1 and tuning parameters, find minimizer of penalized SSE
  
  
  int back_fail=0;
  
  VectorXd c=c0;    
  List obj_list;
  double obj_val;
  List obj_list_new;
  int change_method_ind=ini_method_ind; 
  double delta=0;
  //Rcout << "-----------update C --------------"<< std::endl;
  for(int count=0; count<10; count++)
  {
    //Rcout << "--------"<< "count="<< count<<"-------"<< std::endl;
    List drvc_ret=c_getDrv_c(a, B, V, c, seg_end_index_C,
                             delta_Y, Z, Phi, mu_vec, J_s0,
                             J_u0, J_s2, J_u2, lambdas);
    VectorXd dPLdc_vec=drvc_ret["dPLdc"];
    MatrixXd d2PLd2c=drvc_ret["d2PLd2c"];
    MatrixXd d2PLd2c_part=drvc_ret["d2PLd2c_part"];
    obj_list=drvc_ret["obj_list"];
    obj_val=obj_list["obj_val"];
    VectorXd dc, c_new; 
    //Rcout << "obj_val="<< obj_val  << std::endl; 
    //Rcout << "delta="<< delta << std::endl; 
    //Rcout << "the norm of dPLdc="<< dPLdc_vec.norm() << std::endl; 
    MatrixXd id_mtx=MatrixXd::Identity(d2PLd2c.rows(), d2PLd2c.cols());
    
    
    if(change_method_ind==0)
    {
      dc=-(d2PLd2c_part+1e-5*id_mtx).householderQr().solve(dPLdc_vec);
      
      int dd=dc.array().isNaN().any()*1; 
      if(dd==1)
      {
        dc=-d2PLd2c.householderQr().solve(dPLdc_vec);
        change_method_ind=1;
      }
      c_new=c+dc;
    }
    else
    {
      dc=-(d2PLd2c+delta*id_mtx).householderQr().solve(dPLdc_vec);
      c_new=c+dc;
    }
    obj_list_new=c_obj_val(a, B, V, c_new, seg_end_index_C, delta_Y, Z, Phi, mu_vec, J_s0,
                           J_u0, J_s2, J_u2, lambdas);
    
    double  obj_val_new=obj_list_new["obj_val"];
    //Rcout << "obj_val_new=" << obj_val_new << std::endl;
    if(isnan(obj_val_new)==1)
    {
      obj_val_new=1e30;
    }
    
    
    back_fail=0;
    int been_back=0;
    if(obj_val<obj_val_new)
    { 
      
      for(int k=0; k<4; k++)
      {
        dc=dc/3;
        c_new=c+dc;
        List obj_list_tmp=c_obj_val(a, B, V, c_new, seg_end_index_C, delta_Y, Z, Phi, mu_vec, J_s0,
                                    J_u0, J_s2, J_u2, lambdas);
        double obj_val_tmp=obj_list_tmp["obj_val"];
        //Rcout << "k="<< k<<", "<<"obj_val_tmp=" << obj_val_tmp << std::endl;
        been_back=k+1;
        if(obj_val>obj_val_tmp)
        {
          back_fail=0;
          obj_val_new=obj_val_tmp;
          obj_list_new=obj_list_tmp; 
          break;
        }
        back_fail=1;
        
      }
    }
    //Rcout << "obj_val_new=" << obj_val_new << std::endl;
    if((back_fail==0)&(obj_val-obj_val_new>1e-5*obj_val))
    {
      if(((obj_val-obj_val_new<0.01*obj_val) | (been_back>1))&(change_method_ind==0))
      {
        //Rcout << "been_back=" << been_back << std::endl;
        change_method_ind=1;
        EigenSolver<MatrixXd> es(d2PLd2c, false);
        VectorXd eigen_values=es.eigenvalues().real();
        //Rcout << "Iniyially, the eigenvalues of d2PLd2c :" << eigen_values.minCoeff()<<", "<< eigen_values.maxCoeff() << std::endl; 
        double eig_min=eigen_values.minCoeff(), eig_max=eigen_values.maxCoeff();
        if(eig_min<0)
        {
          delta=-2*eig_min;
          if(delta<eig_max*1e-4)
          {
            delta=eig_max*1e-4;
          }
        }
        else
        {
          if(eig_min<eig_max*1e-4)
          {
            delta=eig_max*1e-4;
          }
        }
      }
      if((been_back>0)&(change_method_ind==1))
      {
        //Rcout << "been_back=" << been_back << std::endl; 
        EigenSolver<MatrixXd> es(d2PLd2c, false);
        VectorXd eigen_values=es.eigenvalues().real();
        //Rcout << "the eigenvalues of d2PLd2c :" << eigen_values.minCoeff()<<", "<< eigen_values.maxCoeff() << std::endl; 
        double eig_min=eigen_values.minCoeff(), eig_max=eigen_values.maxCoeff();
        if(eig_min<0)
        {
          delta=-2*eig_min;
          if(delta<eig_max*1e-4)
          {
            delta=eig_max*1e-4;
          }
        }
        else
        {
          if(eig_min<eig_max*1e-4)
          {
            delta=eig_max*1e-4;
          }
        }
      }
      obj_val=obj_val_new;
      obj_list=obj_list_new;
      c=c_new; 
    }
    else
    {
      if(obj_val>obj_val_new)
      {
        obj_val=obj_val_new;
        obj_list=obj_list_new;
        c=c_new; 
      }
      break;
    }
  }
  
  return List::create(_["c"]=c, _["obj_list"]=obj_list);
}



//[[Rcpp::export]]
List c_Min_Pen_Least_Square(VectorXd a0, MatrixXd B0, MatrixXd V0, VectorXd c0, MatrixXd seg_end_index_C,
                            VectorXd delta_Y, MatrixXd Z, MatrixXd Phi,MatrixXd Psi, VectorXd mu_vec, MatrixXd J_s0,
                            MatrixXd J_u0, MatrixXd J_s2, MatrixXd J_u2, 
                            VectorXd lambdas0, int total_iter)
{
  // Given XPhi, Y, basis, ncomp=1 and tuning parameters, find minimizer of penalized SSE
  VectorXd a=a0, c=c0;
  MatrixXd B=B0, V=V0;
  int ncomp=B.cols();
  VectorXd lambdas=lambdas0;
  lambdas(1)=lambdas0(1)/ncomp; 
  List obj_val_list=c_obj_val(a, B, V, c, seg_end_index_C,
                              delta_Y, Z, Phi, mu_vec, J_s0,
                              J_u0, J_s2, J_u2, lambdas);
  
  double obj_val=obj_val_list["obj_val"], obj_val_tmp=1e30, obj_val_all=1e30; 
  List obj_list;
  double scaling; 
  Rcout << "##############################"  << std::endl; 
  Rcout << "lambdas0="<< lambdas0  << std::endl; 
  Rcout << "ncomp="<< ncomp  << "  obj_val=" << obj_val << std::endl;   
  for(int n_all_iter=0; n_all_iter<total_iter; n_all_iter++)
  { 
    
    List drva_ret=c_getDrv_a(a, B, V, c, seg_end_index_C,
                             delta_Y, Z, Phi, mu_vec, J_s0,
                             J_u0, J_s2, J_u2, lambdas);
    obj_list=drva_ret["obj_list"];
    obj_val=obj_list["obj_val"]; 
    VectorXd dPLda_vec=drva_ret["dPLda"];
    MatrixXd d2PLd2a=drva_ret["d2PLd2a"];
    a=a-d2PLd2a.ldlt().solve(dPLda_vec);
    
    
    
    double obj_val_tmp_all=obj_val;  
    int ini_method_ind=0;
    
    for(int n_in_iter=0; n_in_iter<1; n_in_iter++)
    {
      VectorXd scale_vec=(V.transpose()*(J_s0*V)).diagonal();
      for(int jcomp=0; jcomp<ncomp; jcomp++)
      {
        scaling=sqrt(scale_vec(jcomp));
        if(abs(scaling)>1e-10)
        {
          V.col(jcomp)=V.col(jcomp)/scaling; 
          B.col(jcomp)=B.col(jcomp)*scaling; 
        }
      }
      
      List B_ret=update_B(a, B, V, c, seg_end_index_C,
                          delta_Y, Z, Phi, mu_vec, J_s0,
                          J_u0, J_s2, J_u2,  lambdas, ini_method_ind);
      obj_list=B_ret["obj_list"];
      obj_val_tmp=obj_list["obj_val"];
      //Rcout << "##############################"  << std::endl; 
      //Rcout <<  " obj_val after A and before B="<< obj_val << std::endl; 
      double obj_val0=obj_val;
      MatrixXd coef_mat0=Phi*((B*V.transpose())*Psi.transpose());
      MatrixXd BV_mat0=B*V.transpose();
      
      
      if(obj_val>obj_val_tmp)
      { 
        obj_val=obj_val_tmp;
        B=B_ret["B"];
      }
      //Rcout << "obj_val after B=" << obj_val  << std::endl; 
      //Rcout << "##############################"  << std::endl; 
      
      scale_vec=(B.transpose()*(J_u0*B)).diagonal();
      for(int jcomp=0; jcomp<ncomp; jcomp++)
      {
        scaling=sqrt(scale_vec(jcomp));
        if(abs(scaling)>1e-10)
        {
          V.col(jcomp)=V.col(jcomp)*scaling; 
          B.col(jcomp)=B.col(jcomp)/scaling; 
        }
      }
      
      
      //Rcout << "992 start updating V" << std::endl; 
      double obj_val_V=obj_val; 
      for(int n_V_iter=0; n_V_iter<5; n_V_iter++)
      {
        //Rcout << "n_V_iter=" << n_V_iter   << std::endl; 
        
        //Rcout << " obj_val_V=" << obj_val_V << std::endl; 
        for(int jcomp=0; jcomp<ncomp; jcomp++)
        {
          MatrixXd tmp(V.rows(), ncomp-1), tmp_V=V;
          int ind=0;
          for(int k=0; k<ncomp; k++)
          {
            if(k!=jcomp)
            {
              tmp.col(ind)=tmp_V.col(k);
              ind=ind+1;
            }
          }
          MatrixXd tmp1=J_s0*tmp; //ps*(ncomp-1)
          HouseholderQR<MatrixXd> qr(tmp1);
          MatrixXd Q = qr.householderQ(); //orthogonal columns
          MatrixXd Proj=Q.rightCols(tmp.rows()-tmp.cols()); // orthogonal projection matrix
          
          
          
          List V_list=update_vj(jcomp, a, B, V, c, seg_end_index_C,
                                delta_Y, Z, Phi, mu_vec, J_s0,
                                J_u0, J_s2, J_u2, lambdas, Proj);
          List obj_list_new=V_list["obj_list"]; 
          double obj_val_new=obj_list_new["obj_val"];
          //Rcout << " obj_val=" << obj_val << "  obj_val_new=" << obj_val_new 
          //      << " diff=" << obj_val-obj_val_new << std::endl; 
          if(obj_val>obj_val_new)
          {
            obj_val=obj_val_new;
            V=V_list["V"]; 
          }
        }
        
        if(obj_val_V-obj_val>1e-4*obj_val_V)
        { 
          obj_val_V=obj_val;
        }
        else
        {
          
          break;
        }  
      }//end for (n_v_iter)
      
      //update c
      List c_list=update_c(a, B, V, c, seg_end_index_C,
                           delta_Y, Z, Phi, mu_vec, J_s0,
                           J_u0, J_s2, J_u2, lambdas, ini_method_ind);
      
      obj_list=c_list["obj_list"];
      obj_val_tmp=obj_list["obj_val"];
      if(obj_val>obj_val_tmp)
      { 
        obj_val=obj_val_tmp; 
        c=c_list["c"]; 
      } 
      
      if(obj_val_tmp_all-obj_val>1e-5*obj_val_tmp_all)
      { 
        obj_val_tmp_all=obj_val; 
      }
      else
      {
        break;
      }
    }//end for(n_in_iter)
    
    //Rcout << "obj_val_all="<< obj_val_all << " obj_val_all-obj_val=" << obj_val_all-obj_val << std::endl; 
    //Rcout << "(obj_val_all-obj_val)/obj_val_all=" << (obj_val_all-obj_val)/obj_val_all <<std::endl;
    if(obj_val_all-obj_val>1e-5*obj_val_all)
    {
      obj_val_all=obj_val;
    }
    else
    { 
      break;
    }
  }//end for(n_all_iter)
  
  obj_list=c_obj_val(a, B, V, c, seg_end_index_C, delta_Y, Z, Phi, mu_vec, J_s0,
                     J_u0, J_s2,  J_u2,lambdas);
  obj_val=obj_list["obj_val"];
  //Rcout << "obj_val="<< obj_val  << std::endl; 
  return(List::create(_["obj_val"]=obj_val, _["obj_list"]=obj_list, _["B"]=B, _["V"]=V, _["a"]=a, _["c"]=c));
}



//[[Rcpp::export]]
List c_predict(VectorXd Y_hist, int nfsteps, double hf,
               MatrixXd Phi, MatrixXd Psi, VectorXd a, MatrixXd B, MatrixXd V, VectorXd c)
{
  int ncomp=B.cols();
  int nhist=Psi.rows();
  int Nu=Phi.rows(); 
  VectorXd Y_all(nhist+nfsteps);
  VectorXd dy_all=VectorXd::Zero(nhist+nfsteps);
  Y_all.head(nhist)=Y_hist;
  
  MatrixXd PhiBV= Phi * B * V.transpose();
  VectorXd Phia=Phi * a;
  VectorXd Phic=Phi * c;
  
  //t1
  VectorXd Y_prev=Y_hist.reverse();
  VectorXd Zk= Psi.transpose() * Y_prev / nhist;
  VectorXd sig_input=Phic + PhiBV*Zk;
  VectorXd tmp_sig_1=VectorXd::Constant(Nu, 50);
  VectorXd tmp_sig_2=VectorXd::Constant(Nu, -50);
  sig_input=sig_input.array().min(tmp_sig_1.array());
  sig_input=sig_input.array().max(tmp_sig_2.array());
  // Rcout << sig_input << std::endl;
  sig_input=sig_input.array().exp();
  
  VectorXd sigmoid_val_all=sig_input.array()/(1+sig_input.array());
  dy_all(nhist-1)=Phia.dot(sigmoid_val_all) / Nu; 
  Y_all(nhist)=Y_hist(nhist-1) + hf*dy_all(nhist-1);
  
  
  //t2
  Y_prev=Y_all.segment(1,nhist).reverse();
  Zk= Psi.transpose() * Y_prev / nhist;
  sig_input=Phic + PhiBV*Zk;
  sig_input=sig_input.array().min(tmp_sig_1.array());
  sig_input=sig_input.array().max(tmp_sig_2.array()); 
  sig_input=sig_input.array().exp();
  sigmoid_val_all=sig_input.array()/(1+sig_input.array());
  
  dy_all(nhist)=Phia.dot(sigmoid_val_all) / Nu; 
  Y_all(nhist+1)=Y_all(nhist) + 0.5*hf*(3*dy_all(nhist)-dy_all(nhist-1));
  
  //t3
  Y_prev=Y_all.segment(2,nhist).reverse();
  Zk= Psi.transpose() * Y_prev / nhist;
  sig_input=Phic + PhiBV*Zk;
  sig_input=sig_input.array().min(tmp_sig_1.array());
  sig_input=sig_input.array().max(tmp_sig_2.array()); 
  sig_input=sig_input.array().exp();
  sigmoid_val_all=sig_input.array()/(1+sig_input.array());
  
  dy_all(nhist+1)=Phia.dot(sigmoid_val_all) / Nu; 
  Y_all(nhist+2)=Y_all(nhist+1) + hf*(23*dy_all(nhist+1)-16*dy_all(nhist)+5*dy_all(nhist-1))/12;
  
  
  for(int k=3; k<nfsteps; k++)
  {
    
    if(Y_all(nhist+k-1)>1e10)
    {
      Y_all(nhist+k)=2e10;
    }
    else if(Y_all(nhist+k-1)< -1e10)
    {
      Y_all(nhist+k)=-2e10;
    }
    else
    {
      Y_prev=Y_all.segment(k,nhist).reverse();
      Zk= Psi.transpose() * Y_prev / nhist;
      sig_input=Phic + PhiBV*Zk;
      sig_input=sig_input.array().min(tmp_sig_1.array());
      sig_input=sig_input.array().max(tmp_sig_2.array()); 
      sig_input=sig_input.array().exp();
      sigmoid_val_all=sig_input.array()/(1+sig_input.array());
      
      dy_all(nhist+k-1)=Phia.dot(sigmoid_val_all) / Nu; 
      Y_all(nhist+k)=Y_all(nhist+k-1) + hf*(55*dy_all(nhist+k-1)
                                              -59*dy_all(nhist+k-2)
                                              +37*dy_all(nhist+k-3)
                                              -9*dy_all(nhist+k-4))/24;
    }
    
  }
  
  return List::create(_["Y_forecast"]=Y_all, _["dY_currentOn"]=dy_all);
} 

//[[Rcpp::export]]
List c_cal_error(MatrixXd Y_CV_prev_smooth, MatrixXd Y_CV_true, int forward_steps, 
                 double delta_t, MatrixXd Phi, MatrixXd Psi, 
                 VectorXd a, MatrixXd B, MatrixXd V, VectorXd c)
{
  VectorXd error=VectorXd::Zero(Y_CV_true.cols());
  for(int k=0; k<Y_CV_true.cols(); k++)
  {
    List Yc=c_predict(Y_CV_prev_smooth.col(k), forward_steps, delta_t, Phi, Psi,  a,  B,  V, c);
    VectorXd tmp=Yc(0);
    VectorXd tmp1=tmp.tail(forward_steps);
    error(k)=(tmp1-Y_CV_true.col(k)).squaredNorm()/Y_CV_true.rows();
  }
  //Rcout << "error.min()="<< error.minCoeff()<< ", error.max()="<< error.maxCoeff() << std::endl;
  double mean_error=error.mean(), geo_mean_error=exp(error.array().log().mean());
  return List::create(_["error"]=error, _["mean_error"]=mean_error,
                      _["geo_mean_error"]=geo_mean_error);
} 

//[[Rcpp::export]]
List c_select_tuning_parameter(List x_param_list, int max_steps, int forward_steps,
                                  double delta_t, int max_ncomp)
{
  MatrixXd Phi=x_param_list(0);
  MatrixXd Psi=x_param_list(1);
  MatrixXd J_s0=x_param_list(2);
  MatrixXd J_u0=x_param_list(3);
  MatrixXd J_s2=x_param_list(4);
  MatrixXd J_u2=x_param_list(5);
  VectorXd nus=x_param_list(6);
  VectorXd lambdas0=x_param_list(7);
  MatrixXd seg_end_index_C=x_param_list(8);
  VectorXd seg_N=x_param_list(9);
  VectorXd delta_Y=x_param_list(10);
  VectorXd mu_vec=x_param_list(11);
  MatrixXd ZZ=x_param_list(12);
  VectorXd a, a0, a1, a2;
  VectorXd c, c0, c1, c2;
  MatrixXd B, B0, B1, B2;
  MatrixXd V, V0, V1, V2;
  MatrixXd Y_CV_prev_smooth=x_param_list(13);
  MatrixXd Y_CV_true=x_param_list(14);
  List fit, error_ls, fit_opt, tmp_coef;
  int ncomp0=2, ini_max_steps=100;  
  List result_mat_ls(max_ncomp-ncomp0+1), opt_coef_ls(max_ncomp-ncomp0+1);
  VectorXd opt_parms(3), opt_parms_all(3); 
  
  double min_cv_err_all=1e30;
  double min_cv_err=1e30, cur_error;
  for(int k=0; k<4;k++)
  {  
    a=VectorXd::Random(Phi.cols());
    c=VectorXd::Random(Phi.cols());
    B=MatrixXd::Random(Phi.cols(), 2);
    V=MatrixXd::Random(Phi.cols(), 2);
    
    VectorXd Q0=J_s0*V.col(0); 
    V.col(1)=V.col(1)-Q0*(Q0.transpose()*V.col(1))/Q0.squaredNorm(); 
    Rcout << "Initial rep "<< k+1<<" for ncomp="<< ncomp0 << std::endl; 
    fit=c_Min_Pen_Least_Square(a, B, V, c, seg_end_index_C,
                               delta_Y, ZZ, Phi, Psi, mu_vec, J_s0,
                               J_u0, J_s2, J_u2, lambdas0, ini_max_steps);
    a=fit("a");
    c=fit("c");
    B=fit("B");
    V=fit("V"); 
    
    
    error_ls=c_cal_error(Y_CV_prev_smooth, Y_CV_true, forward_steps, delta_t, Phi, Psi,  a,  B,  V, c);
    cur_error=error_ls(2);
    Rcout << "cur_error="<< cur_error << std::endl;
    if(cur_error<min_cv_err)
    {
      a0=a;
      c0=c;
      B0=B;
      V0=V;  
      min_cv_err=cur_error;
      fit_opt=fit;
    }
    Rcout << "min_cv_err="<< min_cv_err << std::endl;
  }
  
  int tot_iter=100;
  for(int ncomp=ncomp0; ncomp<=max_ncomp; ncomp++)
  {  
    List  opt_coef(4), coef_ls(tot_iter);
    VectorXd lambdas=lambdas0; 
    MatrixXd result_mat=MatrixXd::Constant(tot_iter, 4, 1e30);  
    VectorXd lambda_evolution=VectorXd::Zero(tot_iter);  
    int curr_iter=0, flat_ind=0, min_index=0, right_index, left_index;  
    double allow_inc_perc=0.1;
    
    result_mat(curr_iter, 0)=ncomp;
    result_mat(curr_iter, 1)=lambdas(0);
    result_mat(curr_iter, 2)=lambdas(1);
    result_mat(curr_iter, 3)=min_cv_err;
    a=a0;
    c=c0;
    B=B0;
    V=V0;
    tmp_coef=List::create(_["a"]=a,_["c"]=c,_["B"]=B,_["V"]=V);
    coef_ls(curr_iter)=tmp_coef;
    opt_coef=tmp_coef;
    opt_parms(0)=ncomp;
    opt_parms(1)=lambdas(0);
    opt_parms(2)=lambdas(1);
    
    // go through lambda(1)
    
    for(int k=1; k<=3;k++)
    {
      curr_iter=curr_iter+1; 
      lambdas(1)=lambdas(1)*10;
      fit=c_Min_Pen_Least_Square(a, B, V, c, seg_end_index_C,
                                 delta_Y, ZZ, Phi, Psi, mu_vec, J_s0,
                                 J_u0, J_s2, J_u2, lambdas, max_steps);
      
      a=fit("a");
      c=fit("c");
      B=fit("B");
      V=fit("V"); 
      error_ls=c_cal_error(Y_CV_prev_smooth, Y_CV_true, forward_steps, delta_t, Phi, Psi,  a,  B,  V, c);
      cur_error=error_ls(2);
      result_mat(curr_iter, 0)=ncomp;
      result_mat(curr_iter, 1)=lambdas(0);
      result_mat(curr_iter, 2)=lambdas(1);
      result_mat(curr_iter, 3)=cur_error;
      tmp_coef=List::create(_["a"]=a,_["c"]=c,_["B"]=B,_["V"]=V);
      coef_ls(curr_iter)=tmp_coef;
      Rcout << "#############################" << std::endl;
      Rcout << "cur_error="<< cur_error  << std::endl;
      Rcout << "#############################" << std::endl;
      if(cur_error<min_cv_err)
      { 
        min_index=curr_iter; 
        opt_coef=tmp_coef;
        opt_parms(0)=ncomp;
        opt_parms(1)=lambdas(0);
        opt_parms(2)=lambdas(1);
        min_cv_err=cur_error;
        fit_opt=fit;
      }
    }
    right_index=curr_iter; 
    tmp_coef=coef_ls(right_index);
    a=tmp_coef("a"); 
    
    a=a0;
    c=c0;
    B=B0;
    V=V0; 
    lambdas=lambdas0; 
    for(int k=1; k<=3;k++)
    {
      curr_iter=curr_iter+1; 
      lambdas(1)=lambdas(1)*0.1;
      fit=c_Min_Pen_Least_Square(a, B, V, c, seg_end_index_C,
                                 delta_Y, ZZ, Phi, Psi, mu_vec, J_s0,
                                 J_u0, J_s2, J_u2, lambdas, max_steps);
      
      a=fit("a");
      c=fit("c");
      B=fit("B");
      V=fit("V"); 
      error_ls=c_cal_error(Y_CV_prev_smooth, Y_CV_true, forward_steps, delta_t, Phi, Psi,  a,  B,  V, c);
      cur_error=error_ls(2);
      result_mat(curr_iter, 0)=ncomp;
      result_mat(curr_iter, 1)=lambdas(0);
      result_mat(curr_iter, 2)=lambdas(1);
      result_mat(curr_iter, 3)=cur_error;
      tmp_coef=List::create(_["a"]=a,_["c"]=c,_["B"]=B,_["V"]=V);
      coef_ls(curr_iter)=tmp_coef;
      Rcout << "#############################" << std::endl;
      Rcout << "cur_error="<< cur_error  << std::endl;
      Rcout << "#############################" << std::endl;
      if(cur_error<min_cv_err)
      { 
        min_index=curr_iter; 
        opt_coef=tmp_coef;
        opt_parms(0)=ncomp;
        opt_parms(1)=lambdas(0);
        opt_parms(2)=lambdas(1);
        min_cv_err=cur_error;
        fit_opt=fit;
      }
    }
    left_index=curr_iter; 
    tmp_coef=coef_ls(left_index);
    a=tmp_coef("a");  
    Rcout << "#############################" << std::endl;
    Rcout << "left_index="<< left_index << std::endl;
    Rcout << "right_index="<< right_index  << std::endl;
    Rcout << "min_index="<< min_index<< std::endl;
    Rcout << "#############################" << std::endl;
    
    for(int rep=0; rep<5; rep++)
    {
      if(result_mat(left_index, 3)<result_mat(right_index, 3))
      {
        if(result_mat(left_index, 3)-min_cv_err<min_cv_err*allow_inc_perc)
        {
          lambdas(0)=result_mat(left_index, 1);
          lambdas(1)=result_mat(left_index, 2);
          tmp_coef=coef_ls(left_index);
          a=tmp_coef("a"); 
          c=tmp_coef("c");
          B=tmp_coef("B");
          V=tmp_coef("V"); 
          for(int k=1; k<=2;k++)
          {
            curr_iter=curr_iter+1; 
            lambdas(1)=lambdas(1)*0.1;
            fit=c_Min_Pen_Least_Square(a, B, V, c, seg_end_index_C,
                                       delta_Y, ZZ, Phi, Psi, mu_vec, J_s0,
                                       J_u0, J_s2, J_u2, lambdas, max_steps);
            
            a=fit("a");
            c=fit("c");
            B=fit("B");
            V=fit("V"); 
            error_ls=c_cal_error(Y_CV_prev_smooth, Y_CV_true, forward_steps, delta_t, Phi, Psi,  a,  B,  V, c);
            cur_error=error_ls(2);
            result_mat(curr_iter, 0)=ncomp;
            result_mat(curr_iter, 1)=lambdas(0);
            result_mat(curr_iter, 2)=lambdas(1);
            result_mat(curr_iter, 3)=cur_error;
            tmp_coef=List::create(_["a"]=a,_["c"]=c,_["B"]=B,_["V"]=V);
            coef_ls(curr_iter)=tmp_coef;
            Rcout << "#############################" << std::endl;
            Rcout << "cur_error="<< cur_error  << std::endl;
            Rcout << "#############################" << std::endl;
            if(cur_error<min_cv_err)
            { 
              min_index=curr_iter; 
              opt_coef=tmp_coef;
              opt_parms(0)=ncomp;
              opt_parms(1)=lambdas(0);
              opt_parms(2)=lambdas(1);
              min_cv_err=cur_error;
              fit_opt=fit;
            }
          }
          left_index=curr_iter; 
          tmp_coef=coef_ls(left_index);
          a=tmp_coef("a"); 
        }
        else
        {break;}
      }
      else
      {
        if(result_mat(right_index, 3)-min_cv_err<min_cv_err*allow_inc_perc)
        {
          lambdas(0)=result_mat(right_index, 1);
          lambdas(1)=result_mat(right_index, 2);
          tmp_coef=coef_ls(right_index);
          a=tmp_coef("a"); 
          c=tmp_coef("c");
          B=tmp_coef("B");
          V=tmp_coef("V"); 
          for(int k=1; k<=2;k++)
          {
            curr_iter=curr_iter+1; 
            lambdas(1)=lambdas(1)*10;
            fit=c_Min_Pen_Least_Square(a, B, V, c, seg_end_index_C,
                                       delta_Y, ZZ, Phi, Psi, mu_vec, J_s0,
                                       J_u0, J_s2, J_u2, lambdas, max_steps);
            
            a=fit("a");
            c=fit("c");
            B=fit("B");
            V=fit("V"); 
            error_ls=c_cal_error(Y_CV_prev_smooth, Y_CV_true, forward_steps, delta_t, Phi, Psi,  a,  B,  V, c);
            cur_error=error_ls(2);
            result_mat(curr_iter, 0)=ncomp;
            result_mat(curr_iter, 1)=lambdas(0);
            result_mat(curr_iter, 2)=lambdas(1);
            result_mat(curr_iter, 3)=cur_error;
            tmp_coef=List::create(_["a"]=a,_["c"]=c,_["B"]=B,_["V"]=V);
            coef_ls(curr_iter)=tmp_coef;
            Rcout << "#############################" << std::endl;
            Rcout << "cur_error="<< cur_error  << std::endl;
            Rcout << "#############################" << std::endl;
            if(cur_error<min_cv_err)
            { 
              min_index=curr_iter; 
              opt_coef=tmp_coef;
              opt_parms(0)=ncomp;
              opt_parms(1)=lambdas(0);
              opt_parms(2)=lambdas(1);
              min_cv_err=cur_error;
              fit_opt=fit;
            }
          }
          right_index=curr_iter; 
          tmp_coef=coef_ls(right_index);
          a=tmp_coef("a"); 
        }
        else
        {break;}
      }
    }
    
    // go through lambda(0)
    Eigen::Index index;
    result_mat.col(3).minCoeff(&index);
    Rcout << "min index="<< index << std::endl;
    Rcout << "min index error="<< result_mat(index, 3) << std::endl;
    Rcout << "min lambdas="<< result_mat(index, 1)<<' '<< result_mat(index, 2)<< std::endl;
    VectorXd lambdas1=lambdas0;
    lambdas1(0)=result_mat(index, 1);
    lambdas1(1)=result_mat(index, 2);
    
    lambdas=lambdas1;
    tmp_coef=coef_ls(index); 
    a=tmp_coef("a"); 
    c=tmp_coef("c");
    B=tmp_coef("B");
    V=tmp_coef("V"); 
    for(int k=1; k<=3;k++)
    {
      curr_iter=curr_iter+1; 
      lambdas(0)=lambdas(0)*10;
      fit=c_Min_Pen_Least_Square(a, B, V, c, seg_end_index_C,
                                 delta_Y, ZZ, Phi, Psi, mu_vec, J_s0,
                                 J_u0, J_s2, J_u2, lambdas, max_steps);
      
      a=fit("a");
      c=fit("c");
      B=fit("B");
      V=fit("V"); 
      error_ls=c_cal_error(Y_CV_prev_smooth, Y_CV_true, forward_steps, delta_t, Phi, Psi,  a,  B,  V, c);
      cur_error=error_ls(2);
      result_mat(curr_iter, 0)=ncomp;
      result_mat(curr_iter, 1)=lambdas(0);
      result_mat(curr_iter, 2)=lambdas(1);
      result_mat(curr_iter, 3)=cur_error;
      tmp_coef=List::create(_["a"]=a,_["c"]=c,_["B"]=B,_["V"]=V);
      coef_ls(curr_iter)=tmp_coef;
      Rcout << "#############################" << std::endl;
      Rcout << "cur_error="<< cur_error  << std::endl;
      Rcout << "#############################" << std::endl;
      if(cur_error<min_cv_err)
      { 
        min_index=curr_iter; 
        opt_coef=tmp_coef;
        opt_parms(0)=ncomp;
        opt_parms(1)=lambdas(0);
        opt_parms(2)=lambdas(1);
        min_cv_err=cur_error;
        fit_opt=fit;
      }
    }
    right_index=curr_iter; 
    tmp_coef=coef_ls(right_index);
    a=tmp_coef("a"); 
    
    lambdas=lambdas1;
    tmp_coef=coef_ls(index); 
    a=tmp_coef("a"); 
    c=tmp_coef("c");
    B=tmp_coef("B");
    V=tmp_coef("V");  
    for(int k=1; k<=3;k++)
    {
      curr_iter=curr_iter+1; 
      lambdas(0)=lambdas(0)*0.1;
      fit=c_Min_Pen_Least_Square(a, B, V, c, seg_end_index_C,
                                 delta_Y, ZZ, Phi, Psi, mu_vec, J_s0,
                                 J_u0, J_s2, J_u2, lambdas, max_steps);
      
      a=fit("a");
      c=fit("c");
      B=fit("B");
      V=fit("V"); 
      error_ls=c_cal_error(Y_CV_prev_smooth, Y_CV_true, forward_steps, delta_t, Phi, Psi,  a,  B,  V, c);
      cur_error=error_ls(2);
      result_mat(curr_iter, 0)=ncomp;
      result_mat(curr_iter, 1)=lambdas(0);
      result_mat(curr_iter, 2)=lambdas(1);
      result_mat(curr_iter, 3)=cur_error;
      tmp_coef=List::create(_["a"]=a,_["c"]=c,_["B"]=B,_["V"]=V);
      coef_ls(curr_iter)=tmp_coef;
      Rcout << "#############################" << std::endl;
      Rcout << "cur_error="<< cur_error  << std::endl;
      Rcout << "#############################" << std::endl;
      if(cur_error<min_cv_err)
      { 
        min_index=curr_iter; 
        opt_coef=tmp_coef;
        opt_parms(0)=ncomp;
        opt_parms(1)=lambdas(0);
        opt_parms(2)=lambdas(1);
        min_cv_err=cur_error;
        fit_opt=fit;
      }
    }
    left_index=curr_iter; 
    tmp_coef=coef_ls(left_index);
    a=tmp_coef("a");  
    Rcout << "#############################" << std::endl;
    Rcout << "left_index="<< left_index << std::endl;
    Rcout << "right_index="<< right_index  << std::endl;
    Rcout << "min_index="<< min_index<< std::endl;
    Rcout << "#############################" << std::endl;
    
    for(int rep=0; rep<5; rep++)
    {
      if(result_mat(left_index, 3)<result_mat(right_index, 3))
      {
        if(result_mat(left_index, 3)-min_cv_err<min_cv_err*allow_inc_perc)
        {
          lambdas(0)=result_mat(left_index, 1);
          lambdas(1)=result_mat(left_index, 2);
          tmp_coef=coef_ls(left_index);
          a=tmp_coef("a");  
          c=tmp_coef("c");
          B=tmp_coef("B");
          V=tmp_coef("V"); 
          for(int k=1; k<=2;k++)
          {
            curr_iter=curr_iter+1; 
            lambdas(0)=lambdas(0)*0.1;
            fit=c_Min_Pen_Least_Square(a, B, V, c, seg_end_index_C,
                                       delta_Y, ZZ, Phi, Psi, mu_vec, J_s0,
                                       J_u0, J_s2, J_u2, lambdas, max_steps);
            
            a=fit("a");
            c=fit("c");
            B=fit("B");
            V=fit("V"); 
            error_ls=c_cal_error(Y_CV_prev_smooth, Y_CV_true, forward_steps, delta_t, Phi, Psi,  a,  B,  V, c);
            cur_error=error_ls(2);
            result_mat(curr_iter, 0)=ncomp;
            result_mat(curr_iter, 1)=lambdas(0);
            result_mat(curr_iter, 2)=lambdas(1);
            result_mat(curr_iter, 3)=cur_error;
            tmp_coef=List::create(_["a"]=a,_["c"]=c,_["B"]=B,_["V"]=V);
            coef_ls(curr_iter)=tmp_coef;
            Rcout << "#############################" << std::endl;
            Rcout << "cur_error="<< cur_error  << std::endl;
            Rcout << "#############################" << std::endl;
            if(cur_error<min_cv_err)
            { 
              min_index=curr_iter; 
              opt_coef=tmp_coef;
              opt_parms(0)=ncomp;
              opt_parms(1)=lambdas(0);
              opt_parms(2)=lambdas(1);
              min_cv_err=cur_error;
              fit_opt=fit;
            }
          }
          left_index=curr_iter;  
        }
        else
        {break;}
      }
      else
      {
        if(result_mat(right_index, 3)-min_cv_err<min_cv_err*allow_inc_perc)
        {
          lambdas(0)=result_mat(right_index, 1);
          lambdas(1)=result_mat(right_index, 2);
          tmp_coef=coef_ls(right_index);
          a=tmp_coef("a"); 
          c=tmp_coef("c");
          B=tmp_coef("B");
          V=tmp_coef("V"); 
          for(int k=1; k<=2;k++)
          {
            curr_iter=curr_iter+1; 
            lambdas(0)=lambdas(0)*10;
            fit=c_Min_Pen_Least_Square(a, B, V, c, seg_end_index_C,
                                       delta_Y, ZZ, Phi, Psi, mu_vec, J_s0,
                                       J_u0, J_s2, J_u2, lambdas, max_steps);
            
            a=fit("a");
            c=fit("c");
            B=fit("B");
            V=fit("V"); 
            error_ls=c_cal_error(Y_CV_prev_smooth, Y_CV_true, forward_steps, delta_t, Phi, Psi,  a,  B,  V, c);
            cur_error=error_ls(2);
            result_mat(curr_iter, 0)=ncomp;
            result_mat(curr_iter, 1)=lambdas(0);
            result_mat(curr_iter, 2)=lambdas(1);
            result_mat(curr_iter, 3)=cur_error;
            tmp_coef=List::create(_["a"]=a,_["c"]=c,_["B"]=B,_["V"]=V);
            coef_ls(curr_iter)=tmp_coef;
            Rcout << "#############################" << std::endl;
            Rcout << "cur_error="<< cur_error  << std::endl;
            Rcout << "#############################" << std::endl;
            if(cur_error<min_cv_err)
            { 
              min_index=curr_iter; 
              opt_coef=tmp_coef;
              opt_parms(0)=ncomp;
              opt_parms(1)=lambdas(0);
              opt_parms(2)=lambdas(1);
              min_cv_err=cur_error;
              fit_opt=fit;
            }
          }
          right_index=curr_iter; 
          tmp_coef=coef_ls(right_index);
          a=tmp_coef("a"); 
        }
        else
        {break;}
      }
    }
    
    Rcout << "#############################" << std::endl;
    Rcout << "min_cv_err_all="<< min_cv_err_all << std::endl;
    Rcout << "min_cv_err="<< min_cv_err<< std::endl; 
    result_mat_ls(ncomp-ncomp0)=result_mat;
    opt_coef_ls(ncomp-ncomp0)=opt_coef;
    if(min_cv_err_all>min_cv_err)
    { 
      min_cv_err_all=min_cv_err;
      opt_parms_all=opt_parms;
    } 
    Rcout << "min_cv_err_all updated="<< min_cv_err_all << std::endl; 
    Rcout << "#############################" << std::endl;
    if(ncomp<max_ncomp)
    {
      a1=a0;
      c1=c0;
      V1=V0;
      B1=B0;
      MatrixXd tmp_mat(V1.rows(), V1.cols()+1);
      tmp_mat.leftCols(V1.cols())=V1;
      min_cv_err=1e30;
      
      for(int k=0; k<2;k++)
      {   
        a=a1;
        c=c1;
        MatrixXd tmp_B=MatrixXd::Zero(B1.rows(), B1.cols()+1);
        tmp_B.leftCols(B1.cols())=B1;
        B=tmp_B;
        MatrixXd W0=tmp_mat.leftCols(tmp_mat.cols()-1);
        VectorXd W1= VectorXd::Random(W0.rows());
        MatrixXd KV=J_s0*W0;
        HouseholderQR<MatrixXd> qr(KV);
        MatrixXd Q = qr.householderQ();
        MatrixXd Q0=Q.leftCols(W0.cols());
        VectorXd W2=W1-Q0*(Q0.transpose()*W1);
        double W2_norm=(J_s0*W2).dot(W2); 
        tmp_mat.rightCols(1)=W2/sqrt(W2_norm);
        V=tmp_mat; 
        fit=c_Min_Pen_Least_Square(a, B, V, c, seg_end_index_C,
                                   delta_Y, ZZ, Phi, Psi, mu_vec, J_s0,
                                   J_u0, J_s2, J_u2, lambdas0, ini_max_steps);
        a=fit("a");
        c=fit("c");
        B=fit("B");
        V=fit("V"); 
        
        error_ls=c_cal_error(Y_CV_prev_smooth, Y_CV_true, forward_steps, delta_t, Phi, Psi,  a,  B,  V, c);
        cur_error=error_ls(2);
        Rcout << "Initial rep "<< k+1<<" for ncomp="<< B.cols() << std::endl;
        Rcout << "cur_error="<< cur_error << std::endl;
        if(cur_error<min_cv_err)
        {
          a0=a;
          c0=c;
          B0=B;
          V0=V; 
          fit_opt=fit;
          min_cv_err=cur_error;
        }
        Rcout << "min_cv_err="<< min_cv_err << std::endl;
      }
       
    }
    
     
  }
  
  
  
  return List::create(_["minCVerror="]=min_cv_err_all, _["optParms"]=opt_parms_all, _["result_mat_ls"]=result_mat_ls, _["opt_coef_ls"]=opt_coef_ls);
} 



