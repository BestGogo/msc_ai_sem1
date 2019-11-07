package algorithms.planning;

import gridworld.GridWorld;
import policy.Policy;

/**
 * This is First Visit, Constant-α MC
 * Created by dperez on 16/09/15.
 */
public class MonteCarlo extends PlanningAlgorithm
{
    public double alpha;

    public MonteCarlo(GridWorld gw, Policy p, double gamma, double alpha) {
        super(gw, p, gamma);

        //TODO 6: initialize the state-values v(s) for all s in the grid
    }

    @Override
    //Generates the next set of state-value values v(s). It should iterate through all states in the gridworld
    //and update the value of v(s) according to the Monte Carlo update: v(s) = v(s) + alpha * (Return - v(s));
    public void execute() {
        //TODO 7: Update the value of v(s) for all state-values, according to the algorithm's policy and update rule.
    }
}
