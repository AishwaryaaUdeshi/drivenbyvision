fi = open("training_sets.txt", "r")

nums = [round(float(num.strip()), 3) for num in fi]

# assert(len(nums) == 20000000)

fi.close()


p = iter(nums)
to_write = open("training_set.txt", "w")

to_write_2 = open("test_data_set.txt", "w")



for i in range(500):
    for j in range(100):
        for k in range(100):
            if i < 400:
                to_write.write(str(next(p)) + " ")
            else:
                to_write_2.write(str(next(p)) + " ")
        if i < 400:
            to_write.write("\n")
        else:
            to_write_2.write("\n")
    





to_write.close()
to_write_2.close()
