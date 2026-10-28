 ! Copyright (c) 2025 by Thomas A. Manz. Rights reserved.

 MODULE module_read_atomic_population_analysis_information
 
 USE precise
 USE common_variable_declarations
 USE module_atomic_symbol_to_number
 USE module_string_utilities

 IMPLICIT NONE
 
 CONTAINS
 
 SUBROUTINE read_atomic_population_analysis_information()
 !--------------------------------------------------------------------------------------
 ! Reads atomic population analysis results.
 ! Note that the overlap populations should be pre-sorted according to the first column.
 !--------------------------------------------------------------------------------------
 
 CHARACTER(200) :: temp_string
 INTEGER :: input_FID,start_position,end_position,i,z,temp_int
 REAL(kind=dp) :: temp_real


 !Read net atomic charges
   OPEN(newunit=input_FID,file='DDEC6_net_atomic_charges.xyz',iostat=iostat_value,status='old')
   IF (iostat_value /= 0) THEN
      OPEN(newunit=input_FID,file='DDEC6_even_tempered_net_atomic_charges.xyz',iostat=iostat_value,status='old')
   END IF
   IF (iostat_value /= 0) THEN
     WRITE(output_FID,*)'Could not find DDEC6_net_atomic_charges.xyz or &
     &DDEC6_even_tempered_net_atomic_charges.xyz file. Program will terminate.'
     FLUSH(output_FID)
     STOP
   ELSE
     INQUIRE(unit=input_FID,size=fileSize)
     IF (fileSize < 10) THEN
       WRITE(output_FID,*)'Input file DDEC6_net_atomic_charges.xyz appears to be empty. Program will terminate.'
       FLUSH(output_FID)
       STOP
     END IF
   END IF

 !Read periodic vectors (if any)
 READ(input_FID,*)natoms
 ALLOCATE(net_atomic_charge(natoms))
 net_atomic_charge = 0.0_dp  

 READ(input_FID,'(a)')temp_string
 temp_string=StrLowCase(SimpleLine(temp_string))
 vector1 = 0.0_dp
 vector2 = 0.0_dp
 vector3 = 0.0_dp
 IF (temp_string .NE. 'nonperiodic system') THEN
   start_position = index(temp_string,'[{')
   temp_string = trim(adjustl(temp_string(start_position+2:200)))
   end_position = index(temp_string,' ')
   READ(temp_string(1:end_position),*)vector1(1)
   temp_string = adjustl(temp_string(end_position+1:200))
   end_position = index(temp_string,' ')
   READ(temp_string(1:end_position),*)vector1(2)
   temp_string = adjustl(temp_string(end_position+1:200))
   end_position = index(temp_string,'}')
   READ(temp_string(1:end_position),*)vector1(3)
   start_position = index(temp_string,'{')
   temp_string = trim(adjustl(temp_string(start_position+1:200)))
   end_position = index(temp_string,' ')
   READ(temp_string(1:end_position),*)vector2(1)
   temp_string = adjustl(temp_string(end_position+1:200))
   end_position = index(temp_string,' ')
   READ(temp_string(1:end_position),*)vector2(2)
   temp_string = adjustl(temp_string(end_position+1:200))
   end_position = index(temp_string,'}')
   READ(temp_string(1:end_position),*)vector2(3)
   start_position = index(temp_string,'{')
   temp_string = trim(adjustl(temp_string(start_position+1:200)))
   end_position = index(temp_string,' ')
   READ(temp_string(1:end_position),*)vector3(1)
   temp_string = adjustl(temp_string(end_position+1:200))
   end_position = index(temp_string,' ')
   READ(temp_string(1:end_position),*)vector3(2)
   temp_string = adjustl(temp_string(end_position+1:200))
   end_position = index(temp_string,'}')
   READ(temp_string(1:end_position),*)vector3(3)
 END IF 

 ! Set whether each direction is periodic
 periodicA=.false.
 periodicB=.false.
 periodicC=.false.
 IF ((vector1(1) /= 0.0_dp .or. vector1(2) /= 0.0_dp .or. vector1(3) /= 0.0_dp)) THEN
   periodicA=.true.
 END IF
 IF ((vector2(1) /= 0.0_dp .or. vector2(2) /= 0.0_dp .or. vector2(3) /= 0.0_dp)) THEN
   periodicB=.true.
 END IF
 IF ((vector3(1) /= 0.0_dp .or. vector3(2) /= 0.0_dp .or. vector3(3) /= 0.0_dp)) THEN
   periodicC=.true.
 END IF
 vector1 = vector1*bohrperangstrom
 vector2 = vector2*bohrperangstrom
 vector3 = vector3*bohrperangstrom

 !Write parameters to output file 
 WRITE(output_FID,*)'natoms: ',natoms
 
 
 IF ((periodicA .or. periodicB) .or. periodicC) THEN
   WRITE(output_FID,'(a,3f13.6)')' vector1:',vector1
   WRITE(output_FID,'(a,3f13.6)')' vector2:',vector2
   WRITE(output_FID,'(a,3f13.6)')' vector3:',vector3
   FLUSH(output_FID)
 ELSE
   WRITE(output_FID,'(a)') 'Nonperiodic system'
   FLUSH(output_FID)
 END IF

 ALLOCATE(coords(3,natoms))
 ALLOCATE(starting_atomic_dipole(3,natoms))
 ALLOCATE(atomic_number(natoms))
 ALLOCATE(atom_a(natoms)) 
 ALLOCATE(atom_b(natoms)) 
 ALLOCATE(atomic_quadrupole_tensor(6,natoms))

 coords=0.0_dp
 starting_atomic_dipole=0.0_dp
 atomic_number=0
 atom_a=0.0_dp
 atom_b=0.0_dp

 DO i=1,natoms
   READ(input_FID,*)atomic_symbol,coords(1,i),coords(2,i),coords(3,i),net_atomic_charge(i)
   atomic_symbol=trim(adjustl(atomic_symbol))
   CALL atomic_symbol_to_number(atomic_symbol,output_FID,z)
   atomic_number(i)=z
 END DO
 coords=coords*bohrperangstrom

 WRITE(output_FID,*)'  '
 FLUSH(output_FID)

 DO i=1,14
   READ(input_fid,*)temp_string
 END DO
 DO i=1,natoms
   READ(input_fid,*)temp_int,temp_string,temp_real,temp_real,temp_real,temp_real,starting_atomic_dipole(1,i),&
   starting_atomic_dipole(2,i),starting_atomic_dipole(3,i),temp_real,quadrupole_xy,quadrupole_xz,quadrupole_yz,&
   quadrupole_x2minusy2,quadrupole_3z2minusr2
   atomic_quadrupole_tensor(1,i)=0.5_dp*(quadrupole_x2minusy2 - onethird*quadrupole_3z2minusr2)
   atomic_quadrupole_tensor(2,i)=quadrupole_xy
   atomic_quadrupole_tensor(3,i)=quadrupole_xz
   atomic_quadrupole_tensor(4,i)=0.5_dp*(-quadrupole_x2minusy2 - onethird*quadrupole_3z2minusr2)
   atomic_quadrupole_tensor(5,i)=quadrupole_yz
   atomic_quadrupole_tensor(6,i)=onethird*quadrupole_3z2minusr2
 END DO

 !Read in a and b parameters
 DO i=1,2
   READ(input_fid,*)temp_string
 END DO
 DO i=1,natoms
   READ(input_fid,*)temp_int,temp_string,temp_real,temp_real,temp_real,atom_a(i),atom_b(i)
 END DO
 CLOSE(input_fid)

 END SUBROUTINE read_atomic_population_analysis_information

 SUBROUTINE read_overlap_populations()
 !--------------------------------------------------------------
 ! Reads information regarding the atomic overlap.
 !--------------------------------------------------------------
 IMPLICIT NONE
 
 CHARACTER(200) :: temp_string
 INTEGER :: input_FID,j,k,position,trans_a,trans_b,trans_c
 INTEGER,ALLOCATABLE,DIMENSION(:) :: temp_count_for_atoms
 REAL(kind=dp) :: overlap_population,bond_vector(3)
  
 OPEN(newunit=input_FID,file='overlap_populations_and_contact_exchanges.txt',iostat=iostat_value,status='old')
 IF (iostat_value /= 0) OPEN(newunit=input_FID,file='overlap_populations.txt',iostat=iostat_value,status='old')
 IF (iostat_value /= 0) THEN
   WRITE(output_FID,*)'Could not find overlap_populations_and_contact_exchanges.txt or overlap_populations.txt file. &
   &Program will terminate.'
   FLUSH(output_FID)
   STOP
 ELSE
   INQUIRE(unit=input_FID,size=fileSize)
   IF (fileSize < 10) THEN
     WRITE(output_FID,*)'Input file overlap_populations_and_contact_exchanges.txt or overlap_populations.txt appears to be empty.&
     & Program will terminate.'
     FLUSH(output_FID)
     STOP
   END IF
 END IF
 
 READ(input_FID,*)num_QDR_pairs

 WRITE(output_FID,*)' '
 WRITE(output_FID,*)'num_overlap_values: ',num_QDR_pairs
 WRITE(output_FID,*)' '
 FLUSH(output_FID)

 ALLOCATE(overlap_array_real_variables(8,num_QDR_pairs))
 ALLOCATE(overlap_array_int_variables(5,num_QDR_pairs))
 ALLOCATE(temp_count_for_atoms(natoms))
 ALLOCATE(start_value_for_atoms(natoms+1))

 overlap_array_real_variables=0.0_dp
 overlap_array_int_variables=0 
 temp_count_for_atoms = 0

 READ(input_FID,*)temp_string
 READ(input_FID,*)temp_string
 DO j=1,num_QDR_pairs
   READ(input_FID,*) atom1
   temp_count_for_atoms(atom1) = temp_count_for_atoms(atom1) + 1
 END DO
 CLOSE(input_FID)

 start_value_for_atoms = 0
 start_value_for_atoms(1) = 1
 DO atom1=1,natoms
    start_value_for_atoms(atom1+1) = start_value_for_atoms(atom1) + temp_count_for_atoms(atom1)
 END DO

 temp_count_for_atoms = 0

 OPEN(newunit=input_FID,file='overlap_populations_and_contact_exchanges.txt',iostat=iostat_value,status='old')
 IF (iostat_value /= 0) OPEN(newunit=input_FID,file='overlap_populations.txt',iostat=iostat_value,status='old')
 IF (iostat_value /= 0) THEN
   WRITE(output_FID,*)'Could not find overlap_populations_and_contact_exchanges.txt or overlap_populations.txt file. &
   &Program will terminate.'
   FLUSH(output_FID)
   STOP
 ELSE
   INQUIRE(unit=input_FID,size=fileSize)
   IF (fileSize < 10) THEN
     WRITE(output_FID,*)'Input file overlap_populations_and_contact_exchanges.txt or overlap_populations.txt appears to be empty.&
     & Program will terminate.'
     FLUSH(output_FID)
     STOP
   END IF
 END IF

 READ(input_FID,*)temp_string
 READ(input_FID,*)temp_string
 READ(input_FID,*)temp_string
 DO j=1,num_QDR_pairs
   READ(input_FID,*) atom1,atom2,trans_a,trans_b,trans_c,overlap_population
   i = start_value_for_atoms(atom1) + temp_count_for_atoms(atom1)
   temp_count_for_atoms(atom1) = temp_count_for_atoms(atom1) + 1
   overlap_array_int_variables(1,i) = atom1
   overlap_array_int_variables(2,i) = atom2
   overlap_array_int_variables(3,i) = trans_a
   overlap_array_int_variables(4,i) = trans_b
   overlap_array_int_variables(5,i) = trans_c
   ! The overlap population weighting factor
   overlap_array_real_variables(1,i) = 2.0_dp*tanh(2.0_dp*overlap_population)
   !bond_vector_x
   bond_vector(1) = coords(1,atom2) - coords(1,atom1) &
   + trans_a*vector1(1) + trans_b*vector2(1) + trans_c*vector3(1)
   !bond_vector_y
   bond_vector(2) =  coords(2,atom2) - coords(2,atom1) & 
   + trans_a*vector1(2) + trans_b*vector2(2) + trans_c*vector3(2)
   !bond_vector_z
   bond_vector(3) = coords(3,atom2) - coords(3,atom1) & 
   + trans_a*vector1(3) + trans_b*vector2(3) + trans_c*vector3(3)
   !bond length
   overlap_array_real_variables(5,i) = SQRT(bond_vector(1)*bond_vector(1) &
   + bond_vector(2)*bond_vector(2) + bond_vector(3)*bond_vector(3))
   !bond direction
   overlap_array_real_variables(2:4,i)=(1.0_dp/overlap_array_real_variables(5,i))*bond_vector(1:3)
   !weighted bond direction
   overlap_array_real_variables(6:8,i)=overlap_array_real_variables(1,i)*overlap_array_real_variables(2:4,i) 
 END DO

 CLOSE(input_FID)
 WRITE(output_FID,*)'Finished reading input file overlap_populations.txt'
 WRITE(output_FID,*)' '
 FLUSH(output_FID)

 DEALLOCATE(temp_count_for_atoms)

 END SUBROUTINE read_overlap_populations

 END MODULE module_read_atomic_population_analysis_information
   