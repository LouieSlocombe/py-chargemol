 ! Copyright (c) 2025 by Thomas A. Manz. Rights reserved.

 MODULE common_variable_declarations
 !=====================================================================================
 !Public variables and constants
 !=====================================================================================
 USE precise
 
 IMPLICIT NONE

 !Variables used by different modules
 CHARACTER(2):: atomic_symbol
 CHARACTER(5)  :: zone
 CHARACTER(4) :: prefix
 CHARACTER(8)  :: date
 CHARACTER(10) :: time
 INTEGER :: iostat_value,natoms,output_FID,fileSize,atom1,number_of_threads,num_QDR_pairs,atom2,i,j
 INTEGER :: clock_count,clock_count_rate,clock_count_max,clock_count2,clock_count3 
 INTEGER,DIMENSION(8) :: values
 
 INTEGER,ALLOCATABLE,DIMENSION(:) :: atomic_number,start_value_for_atoms
 INTEGER,ALLOCATABLE,DIMENSION(:,:) :: overlap_array_int_variables
 REAL(kind=dp),ALLOCATABLE,DIMENSION(:) :: net_atomic_charge,atom_a,atom_b,QDR_point_charge
 REAL(kind=dp),ALLOCATABLE,DIMENSION(:,:) :: overlap_array_real_variables,QDR_atomic_dipole,starting_atomic_dipole,coords,&
 atomic_quadrupole_tensor

 REAL(kind=dp) :: vector1(3),vector2(3),vector3(3),V1(3),V2(3),V3(3),seconds

 REAL(kind=dp) :: center_of_mass(3),partial_charge_dipole_x,partial_charge_dipole_y,partial_charge_dipole_z,&
 partial_charge_quadrupole_xy,partial_charge_quadrupole_xz,partial_charge_quadrupole_yz,partial_charge_quadrupole_x2minusy2,&
 partial_charge_quadrupole_3z2minusr2,partial_charge_dipole_magnitude,partial_charge_quadrupole_eigenvals(3),&
 dipole_x,dipole_y,dipole_z,quadrupole_xy,quadrupole_xz,quadrupole_yz,quadrupole_x2minusy2,quadrupole_3z2minusr2,&
 dipole_magnitude,quadrupole_eigenvals(3)
 REAL(kind=dp) :: full_quadrupole_xy,full_quadrupole_xz,full_quadrupole_yz,full_quadrupole_x2minusy2,full_quadrupole_3z2minusr2,&
 full_quadrupole_eigenvals(3)
    
 LOGICAL :: periodicA,periodicB,periodicC,run_parallel,compute_QDR_charges
 
 !===============================================================================================================================
 !Constants
 REAL(kind=dp),PARAMETER :: bohrperangstrom = 1.889725989_dp, onethird=1.0_dp/3.0_dp

 !These are the standard atomic weights of the elements as taken from the NIST.GOV website
 !and published by T.B. Coplen in Atomic Weights of the Elements 1999
 !they include changes reported from the 2001 review in Chem. Int., 23, 179 (2001)
 !these are averaged over natural isotope abundances
 REAL(kind=dp), PARAMETER,DIMENSION(109) :: atomic_weight=[1.00794_dp, 4.002602_dp, 6.941_dp,9.012182_dp,10.811_dp,12.0107_dp,&
 14.0067_dp,15.9994_dp,18.9984032_dp,20.1797_dp,22.989770_dp,24.3050_dp,26.981538_dp,28.0855_dp,30.973761_dp,32.065_dp,35.453_dp,&
 39.948_dp,39.0983_dp,40.078_dp,44.955910_dp,47.867_dp,50.9415_dp,51.9961_dp,54.938049_dp,55.845_dp,58.933200_dp,58.6934_dp,&
 63.546_dp,65.409_dp,69.723_dp,72.64_dp,74.92160_dp,78.96_dp,79.904_dp,83.798_dp,85.4678_dp,87.62_dp,88.90585_dp,91.224_dp,&
 92.90638_dp,95.94_dp,98.0_dp,101.07_dp,102.90550_dp,106.42_dp,107.8682_dp,112.411_dp,114.818_dp,118.710_dp,121.760_dp,&
 127.60_dp,126.90447_dp,131.293_dp,132.90545_dp,137.327_dp,138.9055_dp,140.116_dp,140.90765_dp,144.24_dp,145.0_dp,150.36_dp,&
 151.964_dp,157.25_dp,158.92534_dp,162.500_dp,164.93032_dp,167.259_dp,168.93421_dp,173.04_dp,174.967_dp,178.49_dp,180.9479_dp,&
 183.84_dp,186.207_dp,190.23_dp,192.217_dp,195.078_dp,196.96655_dp,200.59_dp,204.3833_dp,207.2_dp,208.98038_dp,209.0_dp,210.0_dp,&
 222.0_dp,223.0_dp,226.0_dp,227.0_dp,232.0381_dp,231.03588_dp,238.02891_dp,237.0_dp,244.0_dp,243.0_dp,247.0_dp,247.0_dp,251.0_dp,&
 252.0_dp,257.0_dp,258.0_dp,259.0_dp,262.0_dp,261.0_dp,262.0_dp,266.0_dp,264.0_dp,277.0_dp,268.0_dp]

 !===============================================================================================================================
 !Version information
 CHARACTER(51),PARAMETER :: version = 'QDR program version April 9, 2025'
 
 END MODULE common_variable_declarations