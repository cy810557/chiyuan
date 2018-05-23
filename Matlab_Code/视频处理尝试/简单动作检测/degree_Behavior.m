function degree = degree_Behavior(p_moving, p_static)
if p_moving(1) >= p_static(1)
    if p_moving(2) >= p_static(2)
        degree = 270 + atand(abs(p_moving(2)-p_static(2))/abs(p_moving(1) - p_static(1) )); %第四象限
    else
        degree =atand(abs(p_moving(2)-p_static(2))/abs(p_moving(1) - p_static(1) )); %第一象限
    end
else
    if   p_moving(2) >= p_static(2)
        degree =180 + atand(abs(p_moving(2)-p_static(2))/abs(p_moving(1) - p_static(1) )); %第三象限
    else
         degree =180 - atand(abs(p_moving(2)-p_static(2))/abs(p_moving(1) - p_static(1) )); %第二象限
    end
end
end
        