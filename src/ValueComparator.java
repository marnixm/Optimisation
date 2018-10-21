import java.util.Comparator;
import java.util.HashMap;

public class ValueComparator<K, V extends Comparable<V>> implements Comparator<K>{ //TODO from StackOverFlow
	HashMap<K, V> map = new HashMap<K, V>();

	public ValueComparator(HashMap<K, V> map){
		this.map.putAll(map);
	}

	@Override
	public int compare(K s1, K s2) {
		return -map.get(s1).compareTo(map.get(s2)); //descending order
	}
	public static void main(String[] args) {
        System.out.println("comparator");
    }
	
}