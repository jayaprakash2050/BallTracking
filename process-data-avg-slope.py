
def main():
    with open('new-output.csv', 'r') as raw_file:
        processed_file = open('output-goodshots-processed.csv', 'w')
        row = []
        sumOfSlope = 0
        count = 0
        point = None
        for line in raw_file:
            cols = line.split(',')
            if len(cols) == 2:
                cols[0] = float(cols[0].strip())
                cols[1] = float(cols[1].strip())
                if point is not None:
                    print cols[0]
                    print point[0]
                    if cols[1] == point[1] or cols[0] == point[0]:
                        continue
                    sumOfSlope += (cols[1] - point[1]) / (cols[0] - point[0])
                point = (cols[0], cols[1])
                count += 1
            elif len(cols) == 1:
                if cols[0].find(':') == -1 and len(cols[0].strip()) != 0:
                        row.append(str(sumOfSlope/count))
                        row.append(cols[0].strip().replace('\n', ''))
                        processed_file.write(','.join(row) + '\n')
                        sumOfSlope = 0
                        point = None
                        count = 0
                        row = []
    processed_file.flush()
    processed_file.close()

if __name__ == '__main__':
    main()
