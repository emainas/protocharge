#Visualize esp by loading both 1.pose.xyz and esp.xyz and choose "points" for esp.xyz

#Default surface parameters and restrain parameters in TeraChem are:
The default value for this calculation will be a RESP calculation with grid points located on 4 layers of Connolly surface, whose
radii are {1.2, 1,4, 1,6, 1,8 }×vdw_radii, and the density of the grid points on the Connolly Surface is 1.0 point/Å**2. When deriving the atomic charges,
the quadratic restraint alpha * Sum_{k} ((q_{k}**2 + beta**2)**(1/2) − beta) is used, where alpha controls the strength of restraint and 
takes the default value 0.0005 a.u.
and beta controls the tightness near the bottom of the restraint and takes the default value of 0.1e−.

Also I used an extra parameter so that it converges:
esp_grid_dens --> Number of grid points/Å2 (float) Recommendation: Should increase density until the resulting charge is converged 
I used the value of 4.0 (Default is 1.0, but did not converge in my simulation)

#From the Szabo paper: read p. 5 on how the ESP is an observable and see the exact formula for it
#From the classical Griffiths text: read ch. 3.4 on what the multipole expansion is and how it works at the far-field


