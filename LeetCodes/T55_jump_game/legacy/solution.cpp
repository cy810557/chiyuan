// 递归+回溯法（超时））
#include <iostream>
#include <vector>
#include "../vector_utils.cpp"
using namespace std;

void go_through(vector<int> &nums, vector<int>::iterator beg, bool &flag){
    cout<<"current val: "<<*beg<<" ";
    int curr_value = *beg;
    if (curr_value==0) {cout<<"zero step encountered! break.."<<endl;return;}
    if (nums.size()-curr_value-(beg-nums.cbegin())<=1) {cout<<"time to flip..."<<endl;flag=true;return;} //将返回到哪里？
    for (int step=1; step<=curr_value; ++step){
        print(vector<int>(beg, nums.end()));
        if(beg+step<nums.end()) go_through(nums, beg+step, flag);
        else flag=true;
        if (flag==true) break;
    }
}

bool canJump(vector<int> &nums){
    auto beg = nums.begin();
    bool flag=false;
    go_through(nums, beg, flag);
    return flag;
}

int main()
{
    //vector<int> nums = {2,3,1,1,4};
    //vector<int> nums = {3,2,1,0,4};
    vector<int> nums = {2,0,6,9,8,4,5,0,8,9,1,2,9,6,8,8,0,6,3,1,2,2,1,2,6,5,3,1,2,2,6,4,2,4,3,0,0,0,3,8,2,4,0,1,2,0,1,4,6,5,8,0,7,9,3,4,6,6,5,8,9,3,4,3,7,0,4,9,0,9,8,4,3,0,7,7,1,9,1,9,4,9,0,1,9,5,7,7,1,5,8,2,8,2,6,8,2,2,7,5,1,7,9,6};
    bool flag = canJump(nums);
    cout<<(flag ? "can jump!":"cannot jump.")<<endl;
    return 0;
}

