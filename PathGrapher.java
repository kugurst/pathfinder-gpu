import java.io.File;
import java.util.Scanner;

public class PathGrapher
{
	static String testString = "6, 392\n" +
		"human: A, collisions: 0, elements: 7\n" +
		"path: (2,5), (3,5), (4,5), (4,4), (4,3), (3,2), (2,1)\n" +
		"human: B, collisions: 0, elements: 5\n" + 
		"path: (4,5), (4,4), (4,3), (4,2), (4,1)";

	public static void main(String[] args)
	{
		System.out.println(testString);
	}
}