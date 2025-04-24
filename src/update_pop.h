/*	This is a function that takes in Fluxes, the old population, 
 *	timestep size, and updates the new population. 
 * 
 */

std::vector<double>  update_pop( double dt, std::vector<double> F_plus, std::vector<double> F_minus, std::vector<double> y_old){

	std::vector<double> y_new;

	for ( int i = 0; i < size(y_old); i++){
		for ( int j = 0; j < size(y_old); j++){
			y_new[i] = ( y_old[j] + F_plus[j] * dt) / ( 1 + ( F_minus[j] * dt )/ y_old[j]);
		}
	} 
return y_new;
}
