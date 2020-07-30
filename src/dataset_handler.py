import json
import paths_constants

class RedditFile:
    def __init__(self, dataset_file):
        self.dataset_file = dataset_file
        self.__read_file()
    
    def get_summary(self):
        usable_comments = len(self.comment_author)
        usable_comments_percent = '{:.2f}'.format(usable_comments / self.dataset_comments) + '%'

        print('Comments on dataset:', self.dataset_comments)
        print('Comments removing deleted authors:', usable_comments, '({})'.format(usable_comments_percent))
        print('Different authors:', len(self.users))

    def __read_file(self):
        self.dataset_comments = 0 # lines on dataset
        self.users = set()
        self.comment_author = dict()
        self.lines = []

        with open(self.dataset_file) as infile:
            print('Reading file... Depending on the size this may take a while.')

            for line in infile:
                self.dataset_comments += 1

                json_line = json.loads(line)
                l_author = json_line['author'].strip().lower()
        
                # ignore comments from deleted users
                if l_author == '[deleted]':
                    continue
                
                l_id = json_line['id']
                l_parent_id = json_line['parent_id']

                # remove prefix from parent comment
                # https://www.reddit.com/dev/api/
                if '_' in l_parent_id:
                    l_parent_id = l_parent_id.split('_')[1].strip().lower()
                
                self.users.add(l_author)
                self.comment_author[l_id] = l_author
                self.lines.append((l_author, l_parent_id))
    
    def generate_edgelist(self):
        self.edgelist = dict()
        for (user, parent_comment) in self.lines:

            # ignore comments where we don't know the author who received it
            if parent_comment not in self.comment_author:
                continue

            destination_user = self.comment_author[parent_comment]
            edge = (user, destination_user)

            # change edge weight (number of comments from one user to another)
            self.edgelist[edge] = self.edgelist.setdefault(edge, 0) + 1
        
        print('Done. Run save_edgelist() next to save on the processed dataset folder.')

    def save_edgelist(self):
        processed_dataset_path = paths_constants.data_processed_dataset(self.dataset_file.stem)
        processed_dataset_path.mkdir(parents=True, exist_ok=True)
        
        if not hasattr(self, 'edgelist') or not self.edgelist:
            print('Run generate_edgelist() first. Current edgelist is empty.')
            return

        with open(paths_constants.dataset_edgelist(self.dataset_file.stem), 'w') as file:
            for (i,j), w in self.edgelist.items():
                file.write(str(i) + ' ' + str(j) + ' ' + str(w) + '\n')
        
        print('Current edgelist saved.')
