 MODULE module_total_multipole_moment_analysis
 ! Copyright (c) 2022 by Thomas A. Manz. Rights reserved.
   
 USE precise
 USE common_variable_declarations

 IMPLICIT NONE
 
 CONTAINS

 SUBROUTINE compute_center_of_mass()
 !=================================================================================== 
 
 REAL(kind=dp) :: sumvector(3),Mtot
 
 !standard_atomic_weights
 sumvector=0.0_dp
 Mtot=0.0_dp
 DO j=1,natoms
   sumvector(1) = sumvector(1) + atomic_weight(atomic_number(j))*coords(1,j)
   sumvector(2) = sumvector(2) + atomic_weight(atomic_number(j))*coords(2,j)
   sumvector(3) = sumvector(3) + atomic_weight(atomic_number(j))*coords(3,j)
   Mtot = Mtot + atomic_weight(atomic_number(j))
 END DO
 
 center_of_mass=sumvector/Mtot
 
 WRITE(output_FID,'(a,3f10.6)')' center_of_mass: ', center_of_mass
 WRITE(output_FID,*)' '
 FLUSH(output_FID)

 END SUBROUTINE compute_center_of_mass

 
 SUBROUTINE total_multipole_moment_analysis()
 
 REAL(kind=dp) :: x,y,z,charge,temp_quadrupole_matrix(3,3)
 
 IF (((.not. periodicA) .and. (.not. periodicB)) .and. (.not. periodicC)) THEN
   WRITE(output_FID,*)'Since the system is nonperiodic, total multipole moment analysis will be performed.'
   FLUSH(output_FID)
   WRITE(output_FID,*)'Total multipole moments are computed using the center of mass as the origin.'
   FLUSH(output_FID)
   !
   WRITE(output_FID,*)'Dipole and quadrupole moments using the net atomic charges.'
   FLUSH(output_FID)
   WRITE(output_FID,*)'This corresponds to truncatating the distributed multipole expansion at monopole order.'
   FLUSH(output_FID)
   partial_charge_dipole_x = 0.0_dp
   partial_charge_dipole_y = 0.0_dp
   partial_charge_dipole_z = 0.0_dp
   partial_charge_quadrupole_xy = 0.0_dp
   partial_charge_quadrupole_xz = 0.0_dp
   partial_charge_quadrupole_yz = 0.0_dp
   partial_charge_quadrupole_x2minusy2 = 0.0_dp
   partial_charge_quadrupole_3z2minusr2 = 0.0_dp
   DO j = 1,natoms
     x = coords(1,j) - center_of_mass(1)
     y = coords(2,j) - center_of_mass(2)
     z = coords(3,j) - center_of_mass(3)
     charge = QDR_point_charge(j)
     partial_charge_dipole_x = partial_charge_dipole_x + charge*x
     partial_charge_dipole_y = partial_charge_dipole_y + charge*y
     partial_charge_dipole_z = partial_charge_dipole_z + charge*z
     partial_charge_quadrupole_xy = partial_charge_quadrupole_xy + charge*x*y
     partial_charge_quadrupole_xz = partial_charge_quadrupole_xz + charge*x*z
     partial_charge_quadrupole_yz = partial_charge_quadrupole_yz + charge*y*z
     partial_charge_quadrupole_x2minusy2 = partial_charge_quadrupole_x2minusy2 + charge*(x**2 - y**2)
     partial_charge_quadrupole_3z2minusr2 = partial_charge_quadrupole_3z2minusr2 + charge*(3*z*z - (x**2 + y**2 + z**2))
   END DO
   WRITE(output_FID,'(a,f13.6)')' partial_charge_dipole_x= ',partial_charge_dipole_x
   FLUSH(output_FID)
   WRITE(output_FID,'(a,f13.6)')' partial_charge_dipole_y= ',partial_charge_dipole_y
   FLUSH(output_FID)
   WRITE(output_FID,'(a,f13.6)')' partial_charge_dipole_z= ',partial_charge_dipole_z
   FLUSH(output_FID)
   partial_charge_dipole_magnitude = sqrt(partial_charge_dipole_x**2 + partial_charge_dipole_y**2 + partial_charge_dipole_z**2)
   WRITE(output_FID,'(a,f13.6)')' partial_charge_dipole_magnitude',partial_charge_dipole_magnitude
   FLUSH(output_FID)
   WRITE(output_FID,'(a,f13.6)')' partial_charge_quadrupole_xy= ',partial_charge_quadrupole_xy
   FLUSH(output_FID)
   WRITE(output_FID,'(a,f13.6)')' partial_charge_quadrupole_xz= ',partial_charge_quadrupole_xz
   FLUSH(output_FID)
   WRITE(output_FID,'(a,f13.6)')' partial_charge_quadrupole_yz= ',partial_charge_quadrupole_yz
   FLUSH(output_FID)
   WRITE(output_FID,'(a,f13.6)')' partial_charge_quadrupole_x2minusy2= ',partial_charge_quadrupole_x2minusy2
   FLUSH(output_FID)
   WRITE(output_FID,'(a,f13.6)')' partial_charge_quadrupole_3z2minusr2= ',partial_charge_quadrupole_3z2minusr2
   FLUSH(output_FID)
   temp_quadrupole_matrix(1,:)=[0.5_dp*(partial_charge_quadrupole_x2minusy2-partial_charge_quadrupole_3z2minusr2/&
   3.0_dp),partial_charge_quadrupole_xy,partial_charge_quadrupole_xz]
   temp_quadrupole_matrix(2,:)=[partial_charge_quadrupole_xy,0.5_dp*(-partial_charge_quadrupole_x2minusy2-&
   partial_charge_quadrupole_3z2minusr2/3.0_dp),partial_charge_quadrupole_yz]
   temp_quadrupole_matrix(3,:)=[partial_charge_quadrupole_xz,partial_charge_quadrupole_yz,&
   partial_charge_quadrupole_3z2minusr2/3.0_dp]
   partial_charge_quadrupole_eigenvals = eig(temp_quadrupole_matrix)
   !
   WRITE(output_FID,'(a,f13.6)')' Dipole and quadrupole moments using the net atomic charges and the atomic dipoles.'
   FLUSH(output_FID)
   WRITE(output_FID,*)'This corresponds to truncatating the distributed multipole expansion at dipole order.'
   FLUSH(output_FID)
   dipole_x = partial_charge_dipole_x
   dipole_y = partial_charge_dipole_y
   dipole_z = partial_charge_dipole_z   
   quadrupole_xy = partial_charge_quadrupole_xy
   quadrupole_xz = partial_charge_quadrupole_xz
   quadrupole_yz = partial_charge_quadrupole_yz
   quadrupole_x2minusy2 = partial_charge_quadrupole_x2minusy2
   quadrupole_3z2minusr2 = partial_charge_quadrupole_3z2minusr2
   DO j = 1,natoms
     x = coords(1,j) - center_of_mass(1)
     y = coords(2,j) - center_of_mass(2)
     z = coords(3,j) - center_of_mass(3)
     dipole_x = dipole_x + QDR_atomic_dipole(1,j)
     dipole_y = dipole_y + QDR_atomic_dipole(2,j)
     dipole_z = dipole_z + QDR_atomic_dipole(3,j)     
     quadrupole_xy = quadrupole_xy + x*QDR_atomic_dipole(2,j) + y*QDR_atomic_dipole(1,j)
     quadrupole_xz = quadrupole_xz + x*QDR_atomic_dipole(3,j) + z*QDR_atomic_dipole(1,j)
     quadrupole_yz = quadrupole_yz + y*QDR_atomic_dipole(3,j) + z*QDR_atomic_dipole(2,j)
     quadrupole_x2minusy2 = quadrupole_x2minusy2 + 2.0_dp*(x*QDR_atomic_dipole(1,j) - y*QDR_atomic_dipole(2,j))
     quadrupole_3z2minusr2 = quadrupole_3z2minusr2 + 2.0_dp*(2.0_dp*z*QDR_atomic_dipole(3,j) &
     -x*QDR_atomic_dipole(1,j)-y*QDR_atomic_dipole(2,j))
   END DO
   WRITE(output_FID,'(a,f13.6)')' dipole_x= ',dipole_x
   FLUSH(output_FID)
   WRITE(output_FID,'(a,f13.6)')' dipole_y= ',dipole_y
   FLUSH(output_FID)
   WRITE(output_FID,'(a,f13.6)')' dipole_z = ',dipole_z
   FLUSH(output_FID)
   dipole_magnitude = sqrt(dipole_x**2 + dipole_y**2 + dipole_z**2)
   WRITE(output_FID,'(a,f13.6)')' dipole_magnitude= ',dipole_magnitude
   FLUSH(output_FID)
   WRITE(output_FID,'(a,f13.6)')' quadrupole_xy= ',quadrupole_xy
   FLUSH(output_FID)
   WRITE(output_FID,'(a,f13.6)')' quadrupole_xz= ',quadrupole_xz
   FLUSH(output_FID)
   WRITE(output_FID,'(a,f13.6)')' quadrupole_yz= ',quadrupole_yz
   FLUSH(output_FID)
   WRITE(output_FID,'(a,f13.6)')' quadrupole_x2minusy2= ',quadrupole_x2minusy2
   FLUSH(output_FID)
   WRITE(output_FID,'(a,f13.6)')' quadrupole_3z2minusr2 = ',quadrupole_3z2minusr2
   FLUSH(output_FID)
   temp_quadrupole_matrix(1,:)=[0.5_dp*(quadrupole_x2minusy2-quadrupole_3z2minusr2/3.0_dp),quadrupole_xy,quadrupole_xz]
   temp_quadrupole_matrix(2,:)=[quadrupole_xy,0.5_dp*(-quadrupole_x2minusy2-quadrupole_3z2minusr2/3.0_dp),quadrupole_yz]
   temp_quadrupole_matrix(3,:)=[quadrupole_xz,quadrupole_yz,quadrupole_3z2minusr2/3.0_dp]
   quadrupole_eigenvals = eig(temp_quadrupole_matrix)
   !
   WRITE(output_FID,*)'Dipole and quadrupole moments using the net atomic charges, atomic dipoles, and atomic &
   &quadrupoles.'
   FLUSH(output_FID)
   WRITE(output_FID,*)'This corresponds to truncatating the distributed multipole expansion at quadrupole order.'
   FLUSH(output_FID)
   WRITE(output_FID,'(a,f13.6)')' dipole_x= ',dipole_x
   FLUSH(output_FID)
   WRITE(output_FID,'(a,f13.6)')' dipole_y= ',dipole_y
   FLUSH(output_FID)
   WRITE(output_FID,'(a,f13.6)')' dipole_z= ',dipole_z
   FLUSH(output_FID)
   dipole_magnitude = sqrt(dipole_x**2 + dipole_y**2 + dipole_z**2)
   WRITE(output_FID,*) 'dipole_magnitude= ',dipole_magnitude
   FLUSH(output_FID)
   full_quadrupole_xy = quadrupole_xy
   full_quadrupole_xz = quadrupole_xz
   full_quadrupole_yz = quadrupole_yz
   full_quadrupole_x2minusy2 = quadrupole_x2minusy2
   full_quadrupole_3z2minusr2 = quadrupole_3z2minusr2    
   DO j = 1,natoms      
     full_quadrupole_xy = full_quadrupole_xy + atomic_quadrupole_tensor(2,j)
     full_quadrupole_xz = full_quadrupole_xz + atomic_quadrupole_tensor(3,j)
     full_quadrupole_yz = full_quadrupole_yz + atomic_quadrupole_tensor(5,j)
     full_quadrupole_x2minusy2 = full_quadrupole_x2minusy2 + &
     (atomic_quadrupole_tensor(1,j) - atomic_quadrupole_tensor(4,j))
     full_quadrupole_3z2minusr2 = full_quadrupole_3z2minusr2 + 3.0_dp*atomic_quadrupole_tensor(6,j)    
   END DO  
   WRITE(output_FID,'(a,f13.6)')' full_quadrupole_xy= ',full_quadrupole_xy
   FLUSH(output_FID)
   WRITE(output_FID,'(a,f13.6)')' full_quadrupole_xz= ',full_quadrupole_xz
   FLUSH(output_FID)
   WRITE(output_FID,'(a,f13.6)')' full_quadrupole_yz= ',full_quadrupole_yz
   FLUSH(output_FID)
   WRITE(output_FID,'(a,f13.6)')' full_quadrupole_x2minusy2= ',full_quadrupole_x2minusy2
   FLUSH(output_FID)
   WRITE(output_FID,'(a,f13.6)')' full_quadrupole_3z2minusr2 = ',full_quadrupole_3z2minusr2
   FLUSH(output_FID)   
   temp_quadrupole_matrix(1,:)=[0.5_dp*(full_quadrupole_x2minusy2-full_quadrupole_3z2minusr2/3.0_dp),full_quadrupole_xy,&
   full_quadrupole_xz]
   temp_quadrupole_matrix(2,:)=[full_quadrupole_xy,0.5_dp*(-full_quadrupole_x2minusy2-full_quadrupole_3z2minusr2/3.0_dp),&
   full_quadrupole_yz]
   temp_quadrupole_matrix(3,:)=[full_quadrupole_xz,full_quadrupole_yz,full_quadrupole_3z2minusr2/3.0_dp]
   full_quadrupole_eigenvals = eig(temp_quadrupole_matrix)
 ELSE
   WRITE(output_FID,*)'Since the system is periodic, total multipole moment analysis will not be performed.'
 END IF
    
 END SUBROUTINE total_multipole_moment_analysis

 FUNCTION eig(A)
!============================================================
!This function finds and sorts the eigenvalues of a symmetric  
!3x3 matrix.
!http://en.wikipedia.org/wiki/Eigenvalue_algorithm
!============================================================
 
 REAL(kind=dp),INTENT(IN) :: A(3,3)
 REAL(kind=dp) :: I(3,3),p1,q,p2,phi,eig1,eig2,eig3,p,B(3,3),r,eig(3),determinant
 REAL(kind=dp), PARAMETER :: pi = 3.14159265358979323846_dp
 
 !Given a real symmetric 3x3 matrix A, compute the eigenvalues
 p1 = A(1,2)**2 + A(1,3)**2 + A(2,3)**2
 I(1,:) = [1.0_dp,0.0_dp,0.0_dp]
 I(2,:) = [0.0_dp,1.0_dp,0.0_dp]
 I(3,:) = [0.0_dp,0.0_dp,1.0_dp]
 q = (A(1,1)+A(2,2)+A(3,3))/3.0_dp
 IF (p1 < 0.0000000001_dp) THEN
   !A is diagonal.
   eig1 = min(A(1,1),A(2,2),A(3,3))
   eig3 = max(A(1,1),A(2,2),A(3,3))
   eig2 = (q*3.0_dp)-eig1-eig3
 ELSE
   p2 = (A(1,1)-q)**2 + (A(2,2)-q)**2 + (A(3,3)-q)**2+2.0_dp*p1
   p = sqrt(p2/6.0_dp)
   B = (1.0_dp/p) * (A-q*I) 
   determinant=B(1,1)*B(2,2)*B(3,3) + B(1,2)*B(2,3)*B(3,1) + B(1,3)*B(2,1)*B(3,2) &
   -B(1,3)*B(2,2)*B(3,1) - B(1,2)*B(2,1)*B(3,3) - B(1,1)*B(2,3)*B(3,2)
   r = determinant/2.0_dp
   !In exact arithmetic for a symmetric matrix  -1 <= r <= 1
   !but computation error can leave it slightly outside this range.
   IF (r <= -1.0_dp) THEN
      phi = pi/3.0_dp
   ELSE IF (r >= 1.0_dp) THEN
      phi = 0.0_dp
   ELSE
      phi = acos(r)/3.0_dp
   END IF
   !The eigenvalues satisfy eig1 <= eig2 <= eig3
   eig3 = q+2.0_dp*p*cos(phi)
   eig1 = q+2.0_dp*p*cos(phi+(2.0_dp*pi/3.0_dp))
   eig2 = 3.0_dp*q-eig1-eig3     !since trace(A) = eig1 + eig2 + eig3
 END IF
 eig = [eig1,eig2,eig3] 
 
 END FUNCTION eig

 
 END MODULE module_total_multipole_moment_analysis