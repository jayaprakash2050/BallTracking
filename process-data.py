
def main():
    with open('output-goodshots.csv', 'r') as raw_file:
        processed_file = open('output-goodshots-processed.csv', 'w')
        row = []
        sumX = 0
        sumY = 0
        count = 0
        for line in raw_file:
            cols = line.split(',')
            if len(cols) == 2:
                cols[0] = float(cols[0].strip())
                cols[1] = float(cols[1].strip())
                sumX += cols[0]
                sumY += cols[1]
                count += 1
            elif len(cols) == 1:
                if cols[0].find(':') == -1 and len(cols[0].strip()) != 0:
                        row.append(str(sumX/count))
                        row.append(str(sumY/count))
                        row.append(cols[0].strip().replace('\n', ''))
                        processed_file.write(','.join(row) + '\n')
                        count = 0
                        sumX = 0
                        sumY = 0
                        row = []
    processed_file.flush()
    processed_file.close()

if __name__ == '__main__':
    main()