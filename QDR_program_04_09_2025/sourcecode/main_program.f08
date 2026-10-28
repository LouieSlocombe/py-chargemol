 ! Copyright (c) 2025 by Thomas A. Manz. Rights reserved.

 PROGRAM main_program
 !===========================================================
 ! Computes partly-dipole-resorbed charges
 !===========================================================

 USE precise
 USE common_variable_declarations
 USE module_read_atomic_population_analysis_information
 USE module_QDR_point_charges
!$ USE omp_lib

 IMPLICIT NONE

 OPEN(newunit=output_FID,file='QDR_charges_computation_logfile.txt',status='replace')
 WRITE(output_FID,*)' '
 WRITE(output_FID,*)'******************** TIME CONTROL ********************'
 WRITE(output_FID,*)'Entering QDR program.'
 WRITE(output_FID,*) version 
 WRITE(output_FID,*)'Copyright (c) 2025 by Thomas A. Manz. Rights reserved.'
 FLUSH(output_FID)
 CALL system_clock(clock_count,clock_count_rate,clock_count_max)
 WRITE(output_FID,*)'******************************************************'
 WRITE(output_FID,*)' '
 FLUSH(output_FID)
 !Write date in output file
 CALL date_and_time(date,time,zone,values)
 CALL date_and_time(DATE=date,ZONE=zone)
 CALL date_and_time(TIME=time)
 CALL date_and_time(VALUES=values)
 WRITE(output_FID,*) date(1:4),"/",date(5:6),"/",date(7:),"  ",time(1:2),":",time(3:4),":",time(5:6)
 FLUSH(output_FID) 


 !$ number_of_threads = omp_get_max_threads()
 !Write OpenMP information
 run_parallel = .false.
!$ run_parallel = .true.
 WRITE(output_FID,*)' '
 WRITE(output_FID,*)'**************** THREAD INFORMATION ******************'
 IF (run_parallel) THEN
   WRITE(output_FID,*) 'Job running using OpenMP.'
   WRITE(output_FID,*) 'The number of parallel threads is: ',number_of_threads
 ELSE
   WRITE(output_FID,*) 'Job running using serial mode.'
 END IF
 WRITE(output_FID,*)'******************************************************'
 WRITE(output_FID,*) ' '

 !Load atomic population analysis data
 compute_QDR_charges = .TRUE.
 CALL read_atomic_population_analysis_information()
 IF (natoms .EQ. 1) compute_QDR_charges = .FALSE.
 IF (compute_QDR_charges) CALL read_overlap_populations()

 WRITE(output_FID,*)' '
 WRITE(output_FID,*)'******************** TIME CONTROL ********************'
 CALL system_clock(clock_count2,clock_count_rate,clock_count_max)
 IF (clock_count_rate == 0) THEN
   WRITE(output_FID,*)'Processor clock not available, the time calculation wil not be performed.'
 ELSE
   seconds = modulo((clock_count2-clock_count),clock_count_max)/real(clock_count_rate)
   WRITE(output_FID,*)'Input file reading took ',seconds, 'seconds'
   FLUSH(output_FID)
 END IF
 WRITE(output_FID,*)'******************************************************'
 WRITE(output_FID,*)' '

 IF (compute_QDR_charges) THEN
    WRITE(output_FID,*) 'Computing the QDR point charges'
    FLUSH(output_FID)
    CALL generate_QDR_point_charges()
 END IF

 CALL system_clock(clock_count3,clock_count_rate,clock_count_max)

 IF (compute_QDR_charges) THEN
    WRITE(output_FID,*)' '
    WRITE(output_FID,*)'******************** TIME CONTROL ********************'
    IF (clock_count_rate == 0) THEN
      WRITE(output_FID,*)'Processor clock not available, the time calculation wil not be performed.'
    ELSE
      seconds = modulo((clock_count3-clock_count2),clock_count_max)/real(clock_count_rate)
      WRITE(output_FID,*)'Computing the QDR charges took ',seconds, 'seconds'
      FLUSH(output_FID)
    END IF
    WRITE(output_FID,*)'******************************************************'
    WRITE(output_FID,*)' '
 END IF

 WRITE(output_FID,*)' '
 WRITE(output_FID,*)'******************** TIME CONTROL ********************'
 IF (clock_count_rate == 0) THEN
   WRITE(output_FID,*)'Processor clock not available, the time calculation wil not be performed.'
 ELSE
   seconds = modulo((clock_count3-clock_count),clock_count_max)/real(clock_count_rate)
   WRITE(output_FID,*)'Finished QDR program in ',seconds, 'seconds'
   FLUSH(output_FID)
 END IF
 WRITE(output_FID,*)'******************************************************'
 WRITE(output_FID,*)' '
 FLUSH(output_FID) 
 CLOSE(output_FID)
 
 END PROGRAM main_program