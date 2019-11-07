package gridworld;

import algorithms.planning.PlanningAlgorithm;
import utils.JEasyFrame;
import utils.Pair;
import utils.Vector2d;

import javax.swing.*;
import javax.swing.text.Document;
import java.awt.*;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.util.ArrayList;

/**
 * Created by dperez on 26/08/15.
 */
public class GridWorldViewer extends JComponent
{
    public GridWorld gw;
    public PlanningAlgorithm alg;

    public double[][] values;
    public double[][][] qValues;

    private Dimension size;
    public int cellSize = 30;
    private int titlePadding = 15;

    private double min, max;
    private JEasyFrame frame;
    public Point pointMouseClicked;


    public ArrayList<Pair> agentPositions;
    private int prevRow, prevCol;
    private Graphics2D g;

    //This is for printing arrows for qValues. A bit nasty, but it's fixed and expensive to calculate all the time.

    //These are useful, you'll see
    int half = (int) (cellSize * 0.5);
    int third1 = (int) (cellSize * 0.33);
    int third2 = (int) (cellSize * 0.66);
    int quarter = (int) (cellSize*0.25);

    public int polys[][][] = new int[][][]{
            new int[][]{{third2,cellSize,third2},{third1,half,third2}},
            new int[][]{{third1,third1,0},{third1,third2, half}},
            new int[][]{{third1,half,third2},{third1,0,third1}},
            new int[][]{{third1,third2,half},{third2,third2,cellSize}}
    };

    public GridWorldViewer(GridWorld gw, PlanningAlgorithm alg, double[][] values) {
        this.gw = gw;
        this.values = values;
        this.alg = alg;
        pointMouseClicked = null;
        agentPositions = null;
        prevRow = prevCol = -1;

        size = new Dimension(values.length * cellSize, values[0].length * cellSize);

        frame = new JEasyFrame(this, "Gridworld Viewer");

        //frame.addMouseListener(new MouseListener() {
        this.addMouseListener(new MouseListener() {
            @Override
            public void mouseClicked(MouseEvent e) {
                pointMouseClicked = e.getPoint();
            }

            public void mousePressed(MouseEvent e) {
            }

            public void mouseEntered(MouseEvent e) {
            }

            public void mouseReleased(MouseEvent e) {
            }

            public void mouseExited(MouseEvent e) {
            }
        });


    }

    public void paintStateValueFunction(int row, int col, double val)
    {
        double normVal = normalise(val, min, max);
        int green = 255 - (int) (normVal*255);
        green = Math.max(0, Math.min(255,green));

        Color sqColor = new Color(70,green,255);

        g.setColor(Color.black);
        g.drawRect(row * cellSize, col * cellSize, cellSize, cellSize);
        g.setColor(sqColor);
        g.fillRect(row * cellSize, col * cellSize, cellSize, cellSize);
    }

    public void paintActionValueFunction(int row, int col, double[] vals)
    {
        Color sqColors[] = new Color[vals.length];
        double min = Double.MAX_VALUE;
        double max = -Double.MAX_VALUE;

        for(int i = 0; i < vals.length; ++i )
        {
            if(vals[i] < min)
                min = vals[i];
            if(vals[i] > max)
                max = vals[i];
        }

        //Background
        g.setColor(Color.white);
        g.drawRect(row * cellSize, col * cellSize, cellSize, cellSize);

        for(int i = 0; i < vals.length; ++i )
        {
            double normVal = normalise(vals[i], min, max);
            int green = 255 - (int) (normVal*255);
            green = Math.max(0, Math.min(255,green));
            if(green != 0) green = 255;
            Color sqColor = new Color(70,green,255);

            g.setColor(sqColor);
            int[][] points = new int[2][3];

            //Triangle for this action
            points = polys[i];
            g.fillPolygon( new int[]{col * cellSize + points[0][0], col * cellSize + points[0][1],col * cellSize + points[0][2]},
                           new int[]{row * cellSize + points[1][0], row * cellSize + points[1][1],row * cellSize + points[1][2]}, 3);

        }

    }


    /**
     * Main method to paint the game
     * @param gx Graphics object.
     */
    public void paintComponent(Graphics gx)
    {
        g = (Graphics2D) gx;

        //For a better graphics, enable this: (be aware this could bring performance issues depending on your HW & OS).
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

        //g.setColor(Types.LIGHTGRAY);
        g.setColor(Color.black);
        g.fillRect(0, size.height, size.width, size.height);

        for(int i = 0; i < values.length; ++i)
            for(int j = 0; j < values[0].length; ++j)
            {
                //paintStateValueFunction(i, j, values[i][j]);
                if(qValues != null)
                    paintActionValueFunction(i,j,qValues[i][j]);
                else
                    paintStateValueFunction(i, j, values[i][j]);
            }



        int numPos = agentPositions.size();

        if(numPos > 0) {
            Pair p = agentPositions.get(0);
            g.setColor(Color.yellow); //First position
            g.fillOval(p.y * cellSize + quarter, p.x * cellSize + quarter, half, half);

            g.setColor(Color.red);
            for (int i = 1; i < numPos; ++i) {
                p = agentPositions.get(i);
                g.fillOval(p.y * cellSize + quarter, p.x * cellSize + quarter, half, half);
            }
        }

        if(pointMouseClicked != null)
        {
            g.setColor(Color.white);
            g.drawRect(prevCol * cellSize, prevRow * cellSize, cellSize, cellSize);
        }

    }

    /**
     * Paints the grid world.
     */
    public void paint(double min, double max, double values[][], double[][][] qValues)
    {
        this.min = min;
        this.max = max;
        this.values = values;
        this.qValues = qValues;

        if(pointMouseClicked != null)
        {
            int row = (pointMouseClicked.y / cellSize);
            int col = pointMouseClicked.x / cellSize;

            if(row != prevRow || col != prevCol) {

                if (row >= 0 && row < values.length && col >= 0 && col < values[row].length) {

                    frame.setTitle("Gridworld " + values.length + "x" + values[0].length +
                            " [min: " + (int) this.min + ", max" + (int) this.max + "] " +
                            row + " " + col + " " + values[row][col]);

                    String policyStr = "[";
                    double policyProbs[] = new double[this.gw.actions.size()];
                    for (int i = 0; i < policyProbs.length; ++i) {
                        Vector2d act = this.gw.actions.get(i);
                        policyProbs[i] = alg.prob(row, col, act, this.gw.actions);
                        policyStr += policyProbs[i] + ",";
                    }
                    policyStr += "]";

                    System.out.println(policyStr);


                }
            }

            prevRow = row;
            prevCol = col;

        } else {

            frame.setTitle("Gridworld " + values.length + "x" + values[0].length +
                    " [min: " + (int) this.min + ", max" + (int) this.max + "]");
        }

        this.repaint();
    }

    /**
     * Gets the dimensions of the window.
     * @return the dimensions of the window.
     */
    public Dimension getPreferredSize() {
        return size;
    }

    public double normalise(double a_value, double a_min, double a_max)
    {
        if(a_min < a_max)
            return (a_value - a_min)/(a_max - a_min);
        else    // if bounds are invalid, then return 0
            return 0;
    }

}
