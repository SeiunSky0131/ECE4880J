def AddUp(nums, target):
    """
    Input:
    - nums: List[int]
    - target: Int
    
    Returns:
    - List[int]
    """
    answer = []
    # An O(n^2) solution
    for i in range(0,len(nums)):
        for j in range(i, len(nums)):
            if nums[i] + nums[j] == target:
                answer.append(i)
                answer.append(j)
                return answer

# the following is a test function
# if __name__ == '__main__':
#     print(AddUp([1,2,3,4], 7))