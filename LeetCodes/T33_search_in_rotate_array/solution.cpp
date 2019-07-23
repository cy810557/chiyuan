#include<iostream>
#include<vector>
#include "../vector_test.cpp"
using namespace std;

int search(const vector<int> &vi, int target){
    int lo = 0, hi = vi.size();
    while(1<hi-lo){
        int mid = (lo+hi)>>1;
        cout<<lo<<" "<<mid<<" "<<hi<<endl;
        if(vi[lo]<vi[mid]){  //[lo, mid)为有序区间
            if(vi[lo]<=target && target<vi[mid]) hi=mid;
            else lo=mid;
        } else{ //[mid, hi)为有序区间 
            //刚开始写成了target<=vi[hi]，而vi[hi]由于越界产生了一个非常大的数字
            if(vi[mid]<=target && target<=vi[hi-1]) lo=mid;
            else hi=mid;
        }
    }
    return vi[lo]==target ? lo:-1;
}

int main(){
    vector<int> vi = {4,5,6,0,1,2,3};
    print(vi);
    int rst = search(vi, 7);
    cout<<(rst==-1?-1:vi[rst])<<endl;
}
