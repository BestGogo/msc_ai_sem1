package algorithms.control;

import gridworld.GridWorld;
import utils.Pair;
import utils.Vector2d;

import java.util.ArrayList;

/**
 * Created by dperez on 22/09/15.
 */
public class Agent
{
    public Control control;

    public int cRow, cCol;

    public ArrayList<Pair> positions;

    public Agent(){
        cRow = cCol = -1;
        positions = new ArrayList<>();
    }

    public void setControl(GridWorld gw)
    {
        double gamma = 0.9;
        double alpha = 0.1;
        double epsilon = 0.01;

        //this.control = new MonteCarloControl(gw, epsilon, gamma, alpha);
        this.control = new Sarsa(gw, epsilon, gamma, alpha);
        //this.control = new QLearning(gw, epsilon, gamma, alpha);

        this.control.viewer.agentPositions = positions;
    }

    public void train(int totEpisodes)
    {
        long timeNow = System.currentTimeMillis();

        int inc = 1;
        for(int i =0; i < totEpisodes; i+=inc)
        {
            //System.out.println("STEP "  + i + "/"  + totEpisodes );
            this.control.execute(inc);
            this.control.draw();
            waitStep(30);
        }

        long elapsed = System.currentTimeMillis() - timeNow;
        System.out.println("Done in " + elapsed + " ms.");
    }


    public void place(int cRow, int cCol)
    {
        this.cRow = cRow;
        this.cCol = cCol;

        positions.clear();
        positions.add(new Pair(cRow, cCol));
    }

    public Vector2d move()
    {
        Vector2d action = control.policy.moveGreedily(control.qValues, cRow, cCol, control.gridWorld.actions);

        //Update my position
        cRow = Math.max (0, Math.min(cRow + (int) action.y, control.qValues.length-1));
        cCol = Math.max (0, Math.min(cCol + (int) action.x, control.qValues[cRow].length-1));

        positions.add(new Pair(cRow, cCol));

        return action;
    }

    public static void waitStep(int duration) {

        try
        {
            Thread.sleep(duration);
        }
        catch(InterruptedException e)
        {
            e.printStackTrace();
        }
    }

}
