CRAM_MOTOR_COMMANDS = ["#'*backward-transformation*",
 "#'*forward-transformation*",
 "#'*leftward-transformation*",
 "#'*on-transformation*",
 "#'*rightward-transformation*",
 ':BLUE-METAL-PLATE',
 ':BOTTLE',
 ':BOWL',
 ':BREAKFAST-CEREAL',
 ':BUTTERMILK',
 ':CAP',
 ':CEREAL',
 ':CUBE',
 ':CUP',
 ':FORK',
 ':GLASSES',
 ':GLOVE',
 ':KNIFE',
 ':MILK',
 ':MONDAMIN',
 ':MUG',
 ':PLATE',
 ':POT',
 ':RED-METAL-PLATE',
 ':SHOE',
 ':SPATULA',
 ':SPOON',
 ':TRAY',
 ':WEISSWURST',
 'BLUE',
 'GREEN',
 'NIL',
 'POSE-1',
 'POSE-10',
 'POSE-11',
 'POSE-12',
 'POSE-13',
 'POSE-14',
 'POSE-15',
 'POSE-2',
 'POSE-3',
 'POSE-4',
 'POSE-5',
 'POSE-6',
 'POSE-7',
 'POSE-8',
 'POSE-9',
 'RED']

SPECIAL_TOKENS = ["[PAD]"]

ALFRED_action_list = [
    'MoveAhead', 'RotateRight', 'RotateLeft', 
    'LookUp', 'LookDown', 'Pickup', 'Put', 'Open', 
    'Close', 'ToggleOn', 'ToggleOff', 'Slice'
    ]

ALFRED_action_dict = {'MoveAhead': 0,
    'RotateRight': 1,
    'RotateLeft': 2,
    'LookUp': 3,
    'LookDown': 4,
    'Pickup': 5,
    'PickupObject': 5,
    'Put': 6,
    'PutObject': 6,
    'Open': 7,
    'OpenObject': 7,
    'Close': 8,
    'CloseObject': 8,
    'ToggleOn': 9,
    'ToggleObjectOn': 9,
    'ToggleOff': 10,
    'ToggleObjectOff': 10,
    'Slice': 11,
    'SliceObject': 11,
    }

ALFRED_task_type = [
    'look_at_obj_in_light',
    'pick_and_place_simple',
    'pick_and_place_with_movable_recep',
    'pick_clean_then_place_in_recep',
    'pick_cool_then_place_in_recep',
    'pick_heat_then_place_in_recep',
    'pick_two_obj_and_place'
]

def get_vocabulary_dictionary(vocabulary):
    return {word: index for index, word in enumerate(vocabulary)}
    