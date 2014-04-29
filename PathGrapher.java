import java.awt.BorderLayout;
import java.awt.GraphicsEnvironment;
import java.awt.GridLayout;
import java.awt.Rectangle;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;
import java.io.FileNotFoundException;
import java.util.LinkedList;
import java.util.List;
import java.util.Scanner;
import java.util.concurrent.atomic.AtomicBoolean;

import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JPanel;

public class PathGrapher
{
	static AtomicBoolean	start	= new AtomicBoolean(false);

	public static void main(String[] args)
	{
		if (args.length != 2) {
			System.err.println("usage: java PathGrapher <scene.map> <results.txt>");
			System.exit(1);
		}
		// Read in the results
		String results = readResults(args[1]);
		// Read in the map
		int[][] map = readMap(args[0]);
		if (map == null) {
			System.err.println("Improper map file (not found / incorrect format).");
			System.exit(2);
		}
		// Initialize the frame
		JFrame frame = new JFrame("Pathfinder Visualizer");
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		JFrame.setDefaultLookAndFeelDecorated(false);
		frame.getContentPane().setLayout(new BorderLayout());

		// Setup the controls
		JPanel controls = new JPanel(new GridLayout(0, 1, 5, 0));
		JButton simulate = new JButton("Simulate!");
		// Add a listener to the button
		simulate.addActionListener(new ActionListener() {
			@Override
			public void actionPerformed(ActionEvent e)
			{
				start.set(!start.get());
			}
		});
		controls.add(simulate);
		frame.add(controls, BorderLayout.EAST);

		// Setup the canvas
		Rectangle screenSize =
			GraphicsEnvironment.getLocalGraphicsEnvironment().getDefaultScreenDevice()
				.getDefaultConfiguration().getBounds();
		SimulCanvas simult =
			new SimulCanvas(screenSize.width - controls.getPreferredSize().width,
				screenSize.height, results, start, map);
		frame.add(simult, BorderLayout.CENTER);

		// Draw the frame
		frame.pack();
		frame.setLocationRelativeTo(null);
		frame.setVisible(true);
	}

	private static String readResults(String fileName)
	{
		String results = "";
		File file = new File(fileName);
		Scanner in;
		try {
			in = new Scanner(file);
		} catch (FileNotFoundException e) {
			return results;
		}
		while (in.hasNextLine())
			results += in.nextLine() + "\n";
		in.close();
		System.out.println(results);
		return results;
	}

	private static int[][] readMap(String fileName)
	{
		File file = new File(fileName);
		Scanner in;
		try {
			in = new Scanner(file);
		} catch (FileNotFoundException e) {
			return null;
		}
		// Initialize an list to hold all the rows
		List<List<Integer>> list = new LinkedList<List<Integer>>();
		while (in.hasNextLine()) {
			// Parse the line into tokens
			String line = in.nextLine().trim();
			String[] toks = line.split("(\t|\\ |\r)+");
			// Add the tokens to a list
			List<Integer> points = new LinkedList<Integer>();
			for (String tok : toks) {
				// Is it a human?
				if (tok.lastIndexOf(SimulCanvas.HUMCHAR) >= 0)
					points.add(SimulCanvas.THUM);
				// Is it a goal?
				else if (tok.lastIndexOf(SimulCanvas.ENDCHAR) >= 0)
					points.add(SimulCanvas.TEND);
				// Is it an obstacle?
				else if (tok.equals("" + SimulCanvas.OBCHAR)
					|| tok.equals("" + SimulCanvas.BLKCHAR))
					points.add(SimulCanvas.TOBJ);
				// Otherwise, it's a path
				else
					points.add(SimulCanvas.TPATH);
			}
			// Add the list to the overall list
			list.add(points);
		}
		in.close();
		// Get the height of the grid
		int height = list.size();
		// Attempt to get the width of the grid
		if (height < 1)
			return null;
		int width = list.get(0).size();
		// Make an array to hold the values
		int[][] grid = new int[height][width];
		for (int y = 0; y < height; y++)
			for (int x = 0; x < width; x++)
				grid[y][x] = list.get(y).get(x);
		return grid;
	}
}
