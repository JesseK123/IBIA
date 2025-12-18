
# INSERTION SORT (Portfolio Sorting)

def _calculate_portfolio_value(portfolio):
    stocks = portfolio.get("stocks", [])
    total = 0.0
    for s in stocks:
        price = s.get("purchase_price", s.get("price", 0))
        shares = s.get("shares", 1)
        total += price * shares
    return total


def insertion_sort_portfolios(portfolios, key="value"):
    if not portfolios:
        return portfolios
    
    # Create a copy
    portfolios = list(portfolios)
    
    if key == "value":
        for p in portfolios:
            if "value" not in p:
                p["_sort_value"] = _calculate_portfolio_value(p)
        # Use the calculated value for sorting
        sort_key = "_sort_value" if "_sort_value" in portfolios[0] else "value"
    else:
        sort_key = key
    
    # Insertion sort algorithm (descending order)
    for i in range(1, len(portfolios)):
        item = portfolios[i]
        j = i - 1
        
        item_value = item.get(sort_key, 0)

        while j >= 0 and portfolios[j].get(sort_key, 0) < item_value:
            portfolios[j + 1] = portfolios[j]
            j -= 1

        portfolios[j + 1] = item

    return portfolios


# ==== BINARY SEARCH (Stock Symbol Lookup) ====

def binary_search(sorted_list, target):
    """
    Perform iterative binary search on a sorted list.
    
    Precondition: Input list must be sorted in ascending order
    
    Args:
        sorted_list: list of comparable items (sorted ascending)
        target: item to search for
        
    Returns:
        int: Index of target if found, -1 otherwise
    """
    low, high = 0, len(sorted_list) - 1

    while low <= high:
        mid = (low + high) // 2
        mid_value = sorted_list[mid]

        if mid_value == target:
            return mid
        elif mid_value < target:
            low = mid + 1
        else:
            high = mid - 1

    return -1


def binary_search_recursive(sorted_list, target, low=0, high=None):
    """
    Perform recursive binary search on a sorted list.
    Args:
        sorted_list: list of comparable items (sorted ascending)
        target: item to search for
        low: lower bound index (default 0)
        high: upper bound index (default len-1)
        
    Returns:
        int: Index of target if found, -1 otherwise
    """
    if high is None:
        high = len(sorted_list) - 1
    
    # Base case 1: Search space exhausted
    if low > high:
        return -1
    
    mid = (low + high) // 2
    
    # Base case 2: Target found
    if sorted_list[mid] == target:
        return mid
    
    # Recursive cases
    if sorted_list[mid] < target:
        return binary_search_recursive(sorted_list, target, mid + 1, high)
    return binary_search_recursive(sorted_list, target, low, mid - 1)


def find_stock_symbol(all_symbols, target_symbol):
    """
    Search for a stock symbol using binary search.
    
    Time Complexity: O(n log n) for sorting + O(log n) for search
    Space Complexity: O(n) for sorted copy
    
    Args:
        all_symbols: list of stock symbols (unsorted)
        target_symbol: symbol to find
        
    Returns:
        bool: True if symbol exists, False otherwise
    """
    sorted_symbols = sorted(all_symbols)
    index = binary_search(sorted_symbols, target_symbol.upper())
    return index != -1


#SECTION C: MANUAL LINEAR REGRESSION

def manual_linear_regression(y_values):

    #Calculate slope and intercept using least squares method.

    n = len(y_values)
    
    if n == 0:
        return 0, 0
    if n == 1:
        return 0, y_values[0]
    
    xs = list(range(n))
    x_mean = sum(xs) / n
    y_mean = sum(y_values) / n

    numerator = sum((xs[i] - x_mean) * (y_values[i] - y_mean) for i in range(n))
    denominator = sum((xs[i] - x_mean) ** 2 for i in range(n))

    slope = numerator / denominator if denominator != 0 else 0
    intercept = y_mean - slope * x_mean

    return slope, intercept


def predict_price(y_values, days_ahead=365):
    if len(y_values) < 30:
        return None
    
    slope, intercept = manual_linear_regression(list(y_values))
    
    current_price = y_values[-1] if hasattr(y_values, '__getitem__') else list(y_values)[-1]
    future_x = len(y_values) + days_ahead - 1
    predicted_price = slope * future_x + intercept
    
    return {
        "current_price": float(current_price),
        "predicted_price": float(predicted_price),
        "slope": float(slope),
        "intercept": float(intercept),
        "days_ahead": days_ahead
    }


# ==== SECTION D: RECURSIVE ALGORITHMS (Required for IB HL) ====

def recursive_portfolio_value(stocks, index=0):
    """
    Calculate total portfolio value using recursion.
    
    How It Works:
    ------------
    For stocks [A=$100, B=$200, C=$150]:
    
    Call 1: 100 + recursive(index=1)
    Call 2: 200 + recursive(index=2)
    Call 3: 150 + recursive(index=3)
    Call 4: 0 (base case)
    
    Unwinding: 0 → 150 → 350 → 450
    
    Args:
        stocks: list of stock dictionaries
        index: current position in list (default 0)
        
    Returns:
        float: Total value of all stocks
    """
    # Base case: reached end of list
    if index >= len(stocks):
        return 0.0
    
    # Calculate current stock's value
    stock = stocks[index]
    price = stock.get("purchase_price", stock.get("price", 0))
    shares = stock.get("shares", 1)
    current_value = price * shares
    
    # Recursive case: current value + sum of remaining stocks
    return current_value + recursive_portfolio_value(stocks, index + 1)


def recursive_count_stocks(portfolios, index=0):
    """
    Count total stocks across all portfolios recursively.
    
    Time Complexity: O(p) where p = number of portfolios
    Space Complexity: O(p) - call stack depth
    
    Args:
        portfolios: list of portfolio dictionaries
        index: current portfolio index
        
    Returns:
        int: Total count of stocks across all portfolios
    """
    # Base case
    if index >= len(portfolios):
        return 0
    
    current_count = len(portfolios[index].get("stocks", []))
    return current_count + recursive_count_stocks(portfolios, index + 1)


def recursive_find_max_value_portfolio(portfolios, index=0, max_portfolio=None, max_value=0):
    """
    Find portfolio with highest value using recursion.
    
    Time Complexity: O(p × s) where p = portfolios, s = avg stocks
    Space Complexity: O(p) - call stack depth
    
    Args:
        portfolios: list of portfolio dictionaries
        index: current index
        max_portfolio: best portfolio found so far
        max_value: highest value found so far
        
    Returns:
        dict or None: Portfolio with highest total value
    """
    # Base case
    if index >= len(portfolios):
        return max_portfolio
    
    current = portfolios[index]
    current_value = recursive_portfolio_value(current.get("stocks", []))
    
    if current_value > max_value:
        max_portfolio = current
        max_value = current_value
    
    return recursive_find_max_value_portfolio(
        portfolios, index + 1, max_portfolio, max_value
    )


# ==== SECTION E: ABSTRACT DATA TYPES (ADT) ====

class UndoStack:
    """
    Stack implementation for undo functionality (LIFO).
    """
    
    def __init__(self, max_size=50):
        """Initialize empty stack with optional size limit."""
        self._items = []
        self._max_size = max_size
    
    def push(self, action):
        """Add action to top of stack. O(1)"""
        if len(self._items) >= self._max_size:
            self._items.pop(0)
        self._items.append(action)
    
    def pop(self):
        """Remove and return top item. O(1)"""
        if self.is_empty():
            return None
        return self._items.pop()
    
    def peek(self):
        """View top item without removing. O(1)"""
        if self.is_empty():
            return None
        return self._items[-1]
    
    def is_empty(self):
        """Check if stack is empty. O(1)"""
        return len(self._items) == 0
    
    def size(self):
        """Get number of items. O(1)"""
        return len(self._items)
    
    def clear(self):
        """Remove all items. O(1)"""
        self._items = []


class ActionQueue:
    """
    Queue implementation for task processing (FIFO).
    """
    
    def __init__(self):
        """Initialize empty queue."""
        self._items = []
    
    def enqueue(self, item):
        """Add item to back of queue. O(1)"""
        self._items.append(item)
    
    def dequeue(self):
        """Remove and return front item. O(n)"""
        if self.is_empty():
            return None
        return self._items.pop(0)
    
    def peek(self):
        """View front item without removing. O(1)"""
        if self.is_empty():
            return None
        return self._items[0]
    
    def is_empty(self):
        """Check if queue is empty. O(1)"""
        return len(self._items) == 0
    
    def size(self):
        """Get number of items. O(1)"""
        return len(self._items)


# ==== SECTION F: MERGE SORT (Additional Sorting Algorithm) ====

def merge_sort_stocks(stocks, key="symbol"):
    """
    Sort stocks using merge sort algorithm.

    Args:
        stocks: list of stock dictionaries
        key: field to sort by
        
    Returns:
        list: New sorted list
    """
    # Base case
    if len(stocks) <= 1:
        return stocks
    
    # Divide
    mid = len(stocks) // 2
    left_half = stocks[:mid]
    right_half = stocks[mid:]
    
    # Conquer (recursive calls)
    left_sorted = merge_sort_stocks(left_half, key)
    right_sorted = merge_sort_stocks(right_half, key)
    
    # Combine
    return _merge(left_sorted, right_sorted, key)


def _merge(left, right, key):
    """Merge two sorted lists. O(n)"""
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        left_val = left[i].get(key, "")
        right_val = right[j].get(key, "")
        
        if left_val <= right_val:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    
    return result