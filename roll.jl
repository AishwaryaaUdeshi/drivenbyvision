begin

    using Plots

    function gen_road_angle(n, m)
        xs, ys = range(0, 1, n), range(-1, 1, m)
        road, angle = rand(n, m), rand(-1:0.01:1)
        width = round(Int, 0.35m) 
        f(x) = angle * x^2
        for (i, x) in enumerate(xs)
            j = round(Int, (1-m)/(-1-1) * (f(x) - (-1)) + 1 )
            w = width - round(Int, i/5) 
            for k in j-w:j+w
                if 1 <= k <= n
                    road[i, k] = 0.5road[i, k]
                end
            end
        end
        return road, angle
    end

    n, m = 100, 100
	

    function generate_image(n, m, angle)
        xs, ys = range(0, 1, n), range(-1, 1, m)
        road = rand(n, m)
        width = round(Int, 0.35m)
        f(x) = angle * x^2
        for (i, x) in enumerate(xs)
            j = round(Int, (1-m)/(-1-1) * (f(x) - (-1)) + 1 )
            w = width - round(Int, i/5) 
            for k in j-w:j+w
                if 1 <= k <= n
                    road[i, k] = 0.5road[i, k]
                end
            end
        end
        heatmap(road, legend = false, title = "Norm. angle: $angle", c = :grayC)
        
        return road
    end

    all_rds = []

    touch("angles.txt")
    angles = open("angles.txt", "w")
    for i in 1:500
        road, angle = gen_road_angle(n, m)
        push!(all_rds, generate_image(n, m, angle))
        if i <= 400
            write(angles, string(angle))
            write(angles, "\n")
        end
    end

    close(angles)

    touch("training_sets.txt")
    
    data_set = open("training_sets.txt", "w")

    for arr in all_rds
        for arr_row in arr
            for num in arr_row
                write(data_set, string(num) * " ")
            end
            write(data_set, "\n")
        end
    end

    close(data_set)

end