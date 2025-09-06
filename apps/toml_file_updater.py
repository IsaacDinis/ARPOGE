import toml

class TomlFileUpdater:
    def __init__(self, filename):
        self.filename = filename
        self.data = {}

    def add(self, name, value, comment=None):
        # Store value and optional comment as a tuple
        self.data[name] = {'value': value, 'comment': comment}

    def save(self):
        # Format data to include comments on the same line when applicable
        with open(self.filename, 'w') as file:
            for key, val in self.data.items():
                if val['comment']:
                    file.write(f'{key} = {val["value"]}  # {val["comment"]}\n')
                else:
                    file.write(f'{key} = {val["value"]}\n')
        print(f'File saved as {self.filename}')

    def print_toml(self):
        for key, val in self.data.items():
            if val['comment']:
                print(f'{key} = {val["value"]}  # {val["comment"]}')
            else:
                print(f'{key} = {val["value"]}')


if __name__ == '__main__':
    filename = input('Enter the filename: ')
    toml_updater = TomlFileUpdater(filename)

    while True:
        name = input('Enter the name (or type "save" to save and exit): ')
        if name.lower() == 'save':
            toml_updater.save()
            break
        value = input('Enter the value: ')
        comment = input('Enter an optional comment: ')
        toml_updater.add(name, value, comment if comment else None)
        toml_updater.print_toml()