import pdb
def lengthOfLongestSubstring(s: str) -> int:
    _size = len(s)
    before = 0
    count = 1
    after = 1
    while(after != _size):
        while(s[after] not in s[before:after]):
            after += 1;
            if after==_size:
                break
        count = max(after-before, count);
        before += 1;
    return count;

if __name__ == "__main__":
    ss = "abcabcbb"
    print(lengthOfLongestSubstring(ss))
