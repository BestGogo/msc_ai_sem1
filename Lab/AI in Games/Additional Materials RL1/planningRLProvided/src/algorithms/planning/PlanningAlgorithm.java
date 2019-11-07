package algorithms.planning;

import gridworld.GridWorld;
import gridworld.GridWorldViewer;
import policy.Policy;
import utils.Vector2d;

import java.util.ArrayList;

/**
 * Created by dperez on 26/08/15.
 */
public abstract class PlanningAlgorithm {

    //State value of the algorithm (v(s))
    protected double[][] value;

    //Reference to the game/benchmark and its viewer, to draw.
    public GridWorld gridWorld;
    public GridWorldViewer viewer;
    protected double minVal;  //Min value of v(s) for all s. This is only for drawing.
    protected double maxVal;  //Max value of v(s) for all s. This is only for drawing.
    public double gamma; //Gamma for the return function.

    //Policy of the algorithm
    public Policy policy;

    //Number of iterations run.
    public int acumIterations = 0;

    //Constructor of the base class.
    public PlanningAlgorithm(GridWorld gw, Policy p, double gamma) {
        gridWorld = gw;
        value = new double[gw.size][gw.size];
        policy = p;
        viewer = new GridWorldViewer(gridWorld, this, value);
        maxVal = -Double.MAX_VALUE;
        minVal = Double.MAX_VALUE;
        this.gamma = gamma;
    }

    //Returns the v(s), where s is the state reached after applying the action supplied.
    public double getValue(int row, int col, Vector2d action)
    {
        int rowAct = row + (int) action.y;
        int colAct = col + (int) action.x;
        return getValue(rowAct, colAct);
    }

    //Returns V(s) where s is defined as a position (row, col) in the grid.
    public double getValue(int row, int col)
    {
        row = Math.max (0, Math.min(row, value.length-1));
        col = Math.max (0, Math.min(col, value[row].length-1));
        return value[row][col];
    }

    //Gets the return from a certain position (row, col) following the policy given.
    //Define a minimum value for Gamma so iteration doesn't necessarily stop in a terminal state, but after
    //a certain number of steps forward.
    public double getReturn(int row, int col)
    {
        //TODO 1: IMPLEMENT THIS METHOD.
        return 0.0;
    }

    //Returns the probability that, from a state (row,col), the action provided as parameter is actually taken, according to the
    // policy of the algorithm
    public double prob(int row, int col, Vector2d action, ArrayList<Vector2d> actions)
    {
        return policy.prob(value, row, col, action, actions);
    }

    //Indicates the graphic component the values of v(s), minimum and maximum for drawing.
    public void draw()
    {
        viewer.paint(minVal, maxVal, value);
    }

    //Runs the algorithm, to be implemented in the subclasses.
    public abstract void execute();

}
