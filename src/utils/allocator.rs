use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Unique identifier with generation tracking to prevent stale references.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub struct GenerationalId {
    pub index: usize,
    pub generation: u32,
}

impl GenerationalId {
    pub fn new(index: usize, generation: u32) -> Self {
        Self { index, generation }
    }
}

/// Entity identifier wrapper used across the engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub struct EntityId(pub GenerationalId);

impl EntityId {
    pub fn new(index: usize, generation: u32) -> Self {
        Self(GenerationalId::new(index, generation))
    }

    pub fn from_index(index: u32) -> Self {
        Self::new(index as usize, 0)
    }

    pub fn index(&self) -> usize {
        self.0.index
    }

    pub fn generation(&self) -> u32 {
        self.0.generation
    }

    pub fn is_null(&self) -> bool {
        self.0.index == usize::MAX
    }
}

impl Default for EntityId {
    fn default() -> Self {
        Self(GenerationalId::new(usize::MAX, 0))
    }
}

/// Generational arena that hands out stable IDs while preventing use-after-free.
pub struct Arena<T> {
    items: Vec<Option<T>>,
    generations: Vec<u32>,
    free_list: VecDeque<usize>,
}

impl<T> Default for Arena<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Arena<T> {
    pub fn new() -> Self {
        Self {
            items: Vec::new(),
            generations: Vec::new(),
            free_list: VecDeque::new(),
        }
    }

    pub fn insert(&mut self, item: T) -> EntityId {
        if let Some(index) = self.free_list.pop_front() {
            let generation = self.generations[index];
            self.items[index] = Some(item);
            return EntityId::new(index, generation);
        }

        let index = self.items.len();
        self.items.push(Some(item));
        self.generations.push(0);
        EntityId::new(index, 0)
    }

    pub fn get(&self, id: EntityId) -> Option<&T> {
        if self.is_valid(id) {
            self.items.get(id.index()).and_then(|slot| slot.as_ref())
        } else {
            None
        }
    }

    pub fn get_mut(&mut self, id: EntityId) -> Option<&mut T> {
        if self.is_valid(id) {
            self.items.get_mut(id.index()).and_then(|slot| slot.as_mut())
        } else {
            None
        }
    }

    pub fn get2_mut(&mut self, id_a: EntityId, id_b: EntityId) -> Option<(&mut T, &mut T)> {
        if id_a.index() == id_b.index() {
            return None;
        }

        if !self.is_valid(id_a) || !self.is_valid(id_b) {
            return None;
        }

        let (first, second, flipped) = if id_a.index() < id_b.index() {
            (id_a, id_b, false)
        } else {
            (id_b, id_a, true)
        };

        let second_index = second.index();
        if second_index >= self.items.len() {
            return None;
        }

        let (left, right) = self.items.split_at_mut(second_index);
        let first_slot = left
            .get_mut(first.index())
            .and_then(|slot| slot.as_mut())?;
        let second_slot = right.get_mut(0).and_then(|slot| slot.as_mut())?;

        if flipped {
            Some((second_slot, first_slot))
        } else {
            Some((first_slot, second_slot))
        }
    }

    pub fn remove(&mut self, id: EntityId) -> Option<T> {
        if let Some(slot) = self.items.get_mut(id.index()) {
            if slot.is_some() {
                self.generations[id.index()] = self.generations[id.index()].wrapping_add(1);
                self.free_list.push_back(id.index());
            }
            slot.take()
        } else {
            None
        }
    }

    pub fn iter(&self) -> ArenaIter<'_, T> {
        ArenaIter {
            inner: self.items.iter(),
        }
    }

    pub fn iter_mut(&mut self) -> ArenaIterMut<'_, T> {
        ArenaIterMut {
            inner: self.items.iter_mut(),
        }
    }

    pub fn ids(&self) -> impl Iterator<Item = EntityId> + '_ {
        self.items.iter().enumerate().filter_map(|(index, slot)| {
            slot.as_ref()
                .map(|_| EntityId::new(index, self.generations[index]))
        })
    }

    pub fn len(&self) -> usize {
        self.items.iter().filter(|slot| slot.is_some()).count()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn is_valid(&self, id: EntityId) -> bool {
        self.generations
            .get(id.index())
            .copied()
            .map(|gen| gen == id.generation())
            .unwrap_or(false)
    }
}

pub struct ArenaIter<'a, T> {
    inner: std::slice::Iter<'a, Option<T>>,
}

impl<'a, T> Iterator for ArenaIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        for slot in self.inner.by_ref() {
            if let Some(item) = slot.as_ref() {
                return Some(item);
            }
        }
        None
    }
}

pub struct ArenaIterMut<'a, T> {
    inner: std::slice::IterMut<'a, Option<T>>,
}

impl<'a, T> Iterator for ArenaIterMut<'a, T> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        for slot in self.inner.by_ref() {
            if let Some(item) = slot.as_mut() {
                return Some(item);
            }
        }
        None
    }
}
