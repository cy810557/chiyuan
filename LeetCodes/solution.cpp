class Solution {
public:
    ListNode* reverseList(ListNode* head) {
        if(!head) return NULL;
        ListNode *prePtr = NULL, *curPtr = head, *nextPtr = head->next;
        while(nextPtr){
            curPtr->next = prePtr;
            prePtr = curPtr;
            curPtr = nextPtr;
            nextPtr = nextPtr->next;
        }
        curPtr->next = prePtr;
        return curPtr;
    }
};
